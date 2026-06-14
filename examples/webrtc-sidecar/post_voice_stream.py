#!/usr/bin/env python3
"""Post `voice stream` PCM frames to a live WebRTC sidecar call.

This helper connects the current Rust voice daemon stream surface to the
sidecar's local HTTP outbound-audio queue:

    voice stream --raw-output - --sample-rate 48000 --frame-ms 20 ...
      -> POST /calls/{call_id}/audio

It is intentionally stdlib-only so it can be used on machines that have the
`voice` binary but have not installed the optional `aiortc` sidecar extras.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import BinaryIO
from urllib import error, parse, request


CONTRACT_PATH = Path(__file__).resolve().parents[2] / "docs/contracts/webrtc-sidecar-v1.json"
TEMPFAIL_EXIT_CODE = 75


@dataclass(frozen=True)
class AudioContract:
    sample_rate: int
    channels: int
    frame_ms: int
    encoding: str
    frame_bytes: int
    default_drain_bytes: int = 0
    max_outbound_queue_bytes: int = 0
    max_drain_wait_ms: int = 0
    completed_voice_note_command: str = ""
    raw_outbound_pcm_command: str = ""
    raw_inbound_pcm_command: str = ""


class SidecarAudioPostError(RuntimeError):
    """Raised when the sidecar rejects an outbound PCM frame."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        self.retryable = status_code == 429
        super().__init__(
            f"sidecar rejected audio frame ({status_code}): {body}"
        )


class FramePacer:
    """Pace fixed-size PCM frames at the sidecar audio contract interval."""

    def __init__(
        self,
        frame_ms: int,
        *,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        self.interval_s = frame_ms / 1_000
        self.monotonic = monotonic
        self.sleep = sleep
        self.next_frame_at: float | None = None

    def wait(self) -> None:
        now = self.monotonic()
        if self.next_frame_at is None:
            self.next_frame_at = now + self.interval_s
            return

        if now < self.next_frame_at:
            self.sleep(self.next_frame_at - now)
            now = self.monotonic()
        self.next_frame_at = max(self.next_frame_at, now) + self.interval_s


def load_audio_contract(
    path: Path = CONTRACT_PATH,
    voice_bin: str | None = None,
) -> AudioContract:
    if path.exists():
        with path.open(encoding="utf-8") as contract_file:
            contract = json.load(contract_file)
    else:
        contract = load_contract_from_voice(voice_bin)

    audio = contract.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("contract audio section must be an object")

    surfaces = contract.get("voice_surfaces")
    surface_commands = validate_voice_surfaces(surfaces, audio)

    parsed = AudioContract(
        sample_rate=int(audio["sample_rate"]),
        channels=int(audio["channels"]),
        frame_ms=int(audio["frame_ms"]),
        encoding=str(audio["encoding"]),
        frame_bytes=int(audio["frame_bytes"]),
        default_drain_bytes=int(
            audio.get("default_drain_bytes") or audio["frame_bytes"]
        ),
        max_outbound_queue_bytes=int(audio.get("max_outbound_queue_bytes") or 0),
        max_drain_wait_ms=int(audio.get("max_drain_wait_ms") or 0),
        completed_voice_note_command=surface_commands.get("completed_voice_note", ""),
        raw_outbound_pcm_command=surface_commands.get("raw_outbound_pcm", ""),
        raw_inbound_pcm_command=surface_commands.get("raw_inbound_pcm", ""),
    )

    if parsed.sample_rate <= 0:
        raise ValueError("contract sample_rate must be positive")
    if parsed.channels != 1:
        raise ValueError("sidecar currently expects mono audio")
    if parsed.frame_ms <= 0:
        raise ValueError("contract frame_ms must be positive")
    if parsed.encoding != "pcm_s16le":
        raise ValueError("sidecar currently expects pcm_s16le audio")
    if parsed.frame_bytes <= 0 or parsed.frame_bytes % 2 != 0:
        raise ValueError("contract frame_bytes must contain whole s16le samples")
    if parsed.default_drain_bytes <= 0:
        raise ValueError("contract default_drain_bytes must be positive")
    if parsed.default_drain_bytes % parsed.frame_bytes != 0:
        raise ValueError("contract default_drain_bytes must align to WebRTC frames")
    if parsed.max_outbound_queue_bytes <= 0:
        raise ValueError("contract max_outbound_queue_bytes must be positive")
    if parsed.max_outbound_queue_bytes < parsed.frame_bytes:
        raise ValueError("contract max_outbound_queue_bytes must fit one WebRTC frame")
    if parsed.max_drain_wait_ms < 0:
        raise ValueError("contract max_drain_wait_ms must be non-negative")

    return parsed


def validate_voice_surfaces(
    surfaces: object,
    audio: dict[str, object],
) -> dict[str, str]:
    """Validate optional `voice_surfaces` metadata against the PCM contract."""
    if surfaces is None:
        return {}
    if not isinstance(surfaces, dict):
        raise ValueError("contract voice_surfaces section must be an object")

    frame_bytes = int(audio["frame_bytes"])
    encoding = str(audio["encoding"])
    commands: dict[str, str] = {}

    completed = surface_object(surfaces, "completed_voice_note")
    if completed.get("output") != "audio/ogg; codecs=opus":
        raise ValueError("completed_voice_note output must be audio/ogg; codecs=opus")
    if completed.get("transport") != "completed_file":
        raise ValueError("completed_voice_note transport must be completed_file")
    completed_command = surface_command(completed, "completed_voice_note")
    require_command_parts(
        completed_command,
        "completed_voice_note",
        ["voice say", "--format ogg-opus", "--output"],
    )
    commands["completed_voice_note"] = completed_command

    outbound = surface_object(surfaces, "raw_outbound_pcm")
    if outbound.get("output") != encoding:
        raise ValueError("raw_outbound_pcm output must match audio.encoding")
    if outbound.get("transport") != "stdout_pcm_frames":
        raise ValueError("raw_outbound_pcm transport must be stdout_pcm_frames")
    if int(outbound.get("frame_bytes") or 0) != frame_bytes:
        raise ValueError("raw_outbound_pcm frame_bytes must match audio.frame_bytes")
    outbound_command = surface_command(outbound, "raw_outbound_pcm")
    require_command_parts(
        outbound_command,
        "raw_outbound_pcm",
        ["voice stream", "--raw-output", "--sample-rate", "--frame-ms"],
    )
    commands["raw_outbound_pcm"] = outbound_command

    inbound = surface_object(surfaces, "raw_inbound_pcm")
    if inbound.get("input") != encoding:
        raise ValueError("raw_inbound_pcm input must match audio.encoding")
    if inbound.get("transport") != "stdin_pcm_frames":
        raise ValueError("raw_inbound_pcm transport must be stdin_pcm_frames")
    if int(inbound.get("frame_bytes") or 0) != frame_bytes:
        raise ValueError("raw_inbound_pcm frame_bytes must match audio.frame_bytes")
    inbound_command = surface_command(inbound, "raw_inbound_pcm")
    require_command_parts(
        inbound_command,
        "raw_inbound_pcm",
        ["voice stream-transcribe", "--raw-input", "--sample-rate", "--frame-ms"],
    )
    commands["raw_inbound_pcm"] = inbound_command

    return commands


def surface_object(surfaces: dict[str, object], name: str) -> dict[str, object]:
    surface = surfaces.get(name)
    if not isinstance(surface, dict):
        raise ValueError(f"contract voice_surfaces.{name} must be an object")
    return surface


def surface_command(surface: dict[str, object], name: str) -> str:
    command = str(surface.get("command") or "")
    if not command:
        raise ValueError(f"contract voice_surfaces.{name}.command must be non-empty")
    return command


def require_command_parts(command: str, name: str, parts: list[str]) -> None:
    missing = [part for part in parts if part not in command]
    if missing:
        raise ValueError(
            f"contract voice_surfaces.{name}.command is missing: {', '.join(missing)}"
        )


def load_contract_from_voice(voice_bin: str | None = None) -> dict[str, object]:
    command = [voice_bin or os.environ.get("VOICE_BIN", "voice"), "stream-contract"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise RuntimeError(
            "could not load WebRTC contract from docs JSON or `voice stream-contract`"
        ) from exc
    return json.loads(result.stdout)


def sidecar_audio_url(sidecar_url: str, call_id: str) -> str:
    base = sidecar_url.rstrip("/")
    escaped_call_id = parse.quote(call_id, safe="")
    return f"{base}/calls/{escaped_call_id}/audio"


def build_audio_payload(contract: AudioContract, pcm_frame: bytes) -> dict[str, object]:
    if not pcm_frame:
        raise ValueError("pcm_frame must not be empty")
    if len(pcm_frame) % 2 != 0:
        raise ValueError("pcm_frame must contain whole s16le samples")

    return {
        "sample_rate": contract.sample_rate,
        "channels": contract.channels,
        "frame_ms": contract.frame_ms,
        "encoding": contract.encoding,
        "pcm_s16le_base64": base64.b64encode(pcm_frame).decode("ascii"),
    }


def post_audio_frame(
    url: str,
    contract: AudioContract,
    pcm_frame: bytes,
    timeout_s: float,
) -> dict[str, object]:
    payload = json.dumps(build_audio_payload(contract, pcm_frame)).encode("utf-8")
    http_request = request.Request(
        url,
        data=payload,
        method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with request.urlopen(http_request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SidecarAudioPostError(exc.code, body) from exc
    except error.URLError as exc:
        raise RuntimeError(f"failed to post audio frame to sidecar: {exc}") from exc


def stop_voice_process(process: subprocess.Popen[bytes], timeout_s: float = 2.0) -> int:
    """Terminate a still-running `voice stream` child and return its exit code."""
    if process.poll() is not None:
        return int(process.returncode or 0)

    process.terminate()
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def read_exact_or_eof(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_pcm_frames(stream: BinaryIO, frame_bytes: int):
    while True:
        frame = read_exact_or_eof(stream, frame_bytes)
        if not frame:
            return
        if len(frame) < frame_bytes:
            frame += b"\x00" * (frame_bytes - len(frame))
        yield frame


def build_voice_stream_command(args: argparse.Namespace, contract: AudioContract) -> list[str]:
    command = [
        args.voice_bin,
        "stream",
        "--sample-rate",
        str(contract.sample_rate),
        "--frame-ms",
        str(contract.frame_ms),
        "--raw-output",
        "-",
        "--voice",
        args.voice,
        "--speed",
        str(args.speed),
    ]

    if args.markdown:
        command.append("--markdown")
    for substitution in args.sub:
        command.extend(["--sub", substitution])
    if args.sub_file:
        command.extend(["--sub-file", args.sub_file])
    if args.input_file:
        command.extend(["--input-file", args.input_file])
    command.extend(args.text)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("call_id", help="sidecar call_id to send outbound audio to")
    parser.add_argument("text", nargs="*", help="text to stream through voice")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8787")
    parser.add_argument("--voice-bin", default="voice")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--input-file", "-f")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--sub", action="append", default=[])
    parser.add_argument("--sub-file")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.input_file and not args.text:
        parser.error("provide text or --input-file")
    return args


def main() -> int:
    args = parse_args()
    contract = load_audio_contract()
    url = sidecar_audio_url(args.sidecar_url, args.call_id)
    command = build_voice_stream_command(args, contract)

    if not args.quiet:
        print(f"posting {contract.frame_bytes}-byte PCM frames to {url}", file=sys.stderr)

    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None

    frame_count = 0
    byte_count = 0
    post_error: Exception | None = None
    pacer = FramePacer(contract.frame_ms)
    try:
        for frame in iter_pcm_frames(process.stdout, contract.frame_bytes):
            pacer.wait()
            try:
                post_audio_frame(url, contract, frame, args.timeout)
            except Exception as exc:
                post_error = exc
                break
            frame_count += 1
            byte_count += len(frame)
    finally:
        process.stdout.close()

    if post_error is not None:
        return_code = stop_voice_process(process)
        if not args.quiet:
            print(f"{post_error}", file=sys.stderr)
            if isinstance(post_error, SidecarAudioPostError) and post_error.retryable:
                print(
                    "sidecar reported outbound audio backpressure; retry later",
                    file=sys.stderr,
                )
        if isinstance(post_error, SidecarAudioPostError) and post_error.retryable:
            return TEMPFAIL_EXIT_CODE
        return return_code if return_code else 1

    return_code = process.wait()
    if return_code != 0:
        return return_code

    if not args.quiet:
        print(
            f"posted {frame_count} frames ({byte_count} bytes) to {args.call_id}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
