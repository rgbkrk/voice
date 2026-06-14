#!/usr/bin/env python3
"""Drain decoded inbound PCM from a WebRTC sidecar call.

The sidecar exposes decoded WebRTC audio as local HTTP long-poll responses:

    GET /calls/{call_id}/audio?max_bytes=1920&wait_ms=500

This helper writes the returned raw pcm_s16le bytes to a file or stdout. For a
bounded STT smoke test, pipe stdout into:

    voice stream-transcribe --raw-input - --sample-rate 48000 --frame-ms 20
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from pathlib import Path
import sys
import time
from urllib import error, parse, request

from post_voice_stream import AudioContract, load_audio_contract, sidecar_audio_url


def drain_url(sidecar_url: str, call_id: str, max_bytes: int, wait_ms: int) -> str:
    query = parse.urlencode({"max_bytes": max_bytes, "wait_ms": wait_ms})
    return f"{sidecar_audio_url(sidecar_url, call_id)}?{query}"


def validate_drain_shape(contract: AudioContract, max_bytes: int, wait_ms: int) -> None:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if max_bytes % 2 != 0:
        raise ValueError("max_bytes must contain whole s16le samples")
    if max_bytes % contract.frame_bytes != 0:
        raise ValueError(
            f"max_bytes should align to {contract.frame_bytes}-byte WebRTC frames"
        )
    if wait_ms < 0:
        raise ValueError("wait_ms must be non-negative")


def decode_audio_response(body: dict[str, object]) -> bytes:
    payload = str(body.get("pcm_s16le_base64") or "")
    try:
        pcm = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("pcm_s16le_base64 is not valid base64") from exc

    returned_bytes = body.get("returned_bytes")
    if returned_bytes is not None and int(returned_bytes) != len(pcm):
        raise ValueError("returned_bytes does not match decoded PCM length")
    if len(pcm) % 2 != 0:
        raise ValueError("decoded PCM must contain whole s16le samples")
    return pcm


def fetch_audio_chunk(url: str, timeout_s: float) -> bytes:
    try:
        with request.urlopen(url, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"sidecar rejected audio drain ({exc.code}): {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"failed to drain sidecar audio: {exc}") from exc
    return decode_audio_response(body)


def output_stream(path: str):
    if path == "-":
        return sys.stdout.buffer
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("wb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("call_id", help="sidecar call_id to drain inbound audio from")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8787")
    parser.add_argument("--output", "-o", default="-", help="raw PCM output path or '-'")
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--wait-ms", type=int, default=500)
    parser.add_argument(
        "--duration-ms",
        type=int,
        help="stop after this wall-clock duration; omit to run until interrupted",
    )
    parser.add_argument(
        "--stop-after-empty",
        type=int,
        default=0,
        help="stop after N consecutive empty long-poll responses; 0 disables",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_audio_contract()
    max_bytes = args.max_bytes or contract.frame_bytes
    validate_drain_shape(contract, max_bytes, args.wait_ms)
    if args.duration_ms is not None and args.duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")

    url = drain_url(args.sidecar_url, args.call_id, max_bytes, args.wait_ms)
    deadline = (
        time.monotonic() + (args.duration_ms / 1_000)
        if args.duration_ms is not None
        else None
    )

    if not args.quiet:
        print(f"draining inbound PCM from {url}", file=sys.stderr)

    chunks = 0
    bytes_written = 0
    empty_polls = 0
    stream = output_stream(args.output)
    should_close = stream is not sys.stdout.buffer
    try:
        while deadline is None or time.monotonic() < deadline:
            pcm = fetch_audio_chunk(url, args.timeout)
            if not pcm:
                empty_polls += 1
                if args.stop_after_empty and empty_polls >= args.stop_after_empty:
                    break
                continue

            empty_polls = 0
            stream.write(pcm)
            stream.flush()
            chunks += 1
            bytes_written += len(pcm)
    except KeyboardInterrupt:
        pass
    finally:
        if should_close:
            stream.close()

    if not args.quiet:
        print(
            f"drained {chunks} chunks ({bytes_written} bytes) from {args.call_id}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
