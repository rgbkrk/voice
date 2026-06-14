#!/usr/bin/env python3
"""Echo decoded inbound sidecar PCM back to the same WebRTC call.

This helper is a minimal local echo bridge:

    GET  /calls/{call_id}/audio -> decoded inbound pcm_s16le
    POST /calls/{call_id}/audio -> outbound pcm_s16le for the WebRTC track

It proves the first live-call bot shape without involving STT, TTS, Hermes
sessions, WhatsApp Graph calls, or model inference.
"""

from __future__ import annotations

import argparse
import sys
import time

from drain_sidecar_audio import (
    capped_wait_ms,
    drain_url,
    fetch_audio_chunk,
    validate_drain_shape,
)
from post_voice_stream import (
    SidecarAudioPostError,
    load_audio_contract,
    post_audio_frame,
    sidecar_audio_url,
)


TEMPFAIL_EXIT_CODE = 75


def default_max_bytes(configured_max_bytes: int | None, frame_bytes: int) -> int:
    if configured_max_bytes is not None:
        return configured_max_bytes
    return frame_bytes


def echo_chunks(
    *,
    sidecar_url: str,
    call_id: str,
    max_bytes: int | None,
    wait_ms: int,
    duration_ms: int | None,
    stop_after_empty: int,
    timeout_s: float,
    quiet: bool,
) -> int:
    contract = load_audio_contract()
    drain_max_bytes = default_max_bytes(max_bytes, contract.frame_bytes)
    drain_wait_ms = capped_wait_ms(contract, wait_ms)
    validate_drain_shape(contract, drain_max_bytes, drain_wait_ms)
    if duration_ms is not None and duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")

    source_url = drain_url(sidecar_url, call_id, drain_max_bytes, drain_wait_ms)
    target_url = sidecar_audio_url(sidecar_url, call_id)
    deadline = (
        time.monotonic() + (duration_ms / 1_000)
        if duration_ms is not None
        else None
    )

    if not quiet:
        print(f"echoing inbound PCM from {source_url} to {target_url}", file=sys.stderr)

    chunks = 0
    bytes_echoed = 0
    empty_polls = 0
    while deadline is None or time.monotonic() < deadline:
        pcm = fetch_audio_chunk(source_url, timeout_s)
        if not pcm:
            empty_polls += 1
            if stop_after_empty and empty_polls >= stop_after_empty:
                break
            continue

        empty_polls = 0
        try:
            post_audio_frame(target_url, contract, pcm, timeout_s)
        except SidecarAudioPostError as exc:
            if exc.retryable:
                if not quiet:
                    print(
                        "sidecar reported outbound audio backpressure; retry later",
                        file=sys.stderr,
                    )
                return TEMPFAIL_EXIT_CODE
            raise
        chunks += 1
        bytes_echoed += len(pcm)

    if not quiet:
        print(
            f"echoed {chunks} chunks ({bytes_echoed} bytes) for {call_id}",
            file=sys.stderr,
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("call_id", help="sidecar call_id to echo")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8787")
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
    try:
        return echo_chunks(
            sidecar_url=args.sidecar_url,
            call_id=args.call_id,
            max_bytes=args.max_bytes,
            wait_ms=args.wait_ms,
            duration_ms=args.duration_ms,
            stop_after_empty=args.stop_after_empty,
            timeout_s=args.timeout,
            quiet=args.quiet,
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
