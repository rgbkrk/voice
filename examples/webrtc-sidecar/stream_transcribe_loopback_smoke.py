#!/usr/bin/env python3
"""Run sidecar-decoded WebRTC audio through `voice stream-transcribe`.

This smoke proves the inbound live-call media path locally:

    voice stream raw PCM -> local aiortc sender -> WebRTC sidecar
      -> GET /calls/{call_id}/audio decoded PCM
      -> voice stream-transcribe --raw-input ...

It does not contact WhatsApp or the Graph API.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from aiohttp import ClientSession, web

import sidecar


DEFAULT_TEXT = "hello world"
DEFAULT_EXPECT_WORDS = ("hello", "world")


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        return str(path.resolve())

    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return found


def iter_pcm_frames(pcm: bytes, frame_bytes: int = sidecar.FRAME_BYTES):
    if frame_bytes <= 0:
        raise ValueError("frame_bytes must be positive")
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        if len(frame) < frame_bytes:
            frame += b"\x00" * (frame_bytes - len(frame))
        yield frame


class PcmBytesTrack(sidecar.MediaStreamTrack):
    """Finite PCM source followed by silence so the peer connection stays open."""

    kind = "audio"

    def __init__(self, pcm: bytes) -> None:
        super().__init__()
        self.frames = list(iter_pcm_frames(pcm))
        self.index = 0
        self.pts = 0
        self.silence = b"\x00" * sidecar.FRAME_BYTES

    async def recv(self) -> sidecar.av.AudioFrame:
        await asyncio.sleep(sidecar.FRAME_MS / 1_000)
        if self.index < len(self.frames):
            frame_bytes = self.frames[self.index]
            self.index += 1
        else:
            frame_bytes = self.silence

        frame = sidecar.av.AudioFrame(
            format="s16",
            layout="mono",
            samples=sidecar.SAMPLES_PER_FRAME,
        )
        frame.planes[0].update(frame_bytes)
        frame.sample_rate = sidecar.SAMPLE_RATE
        frame.time_base = sidecar.Fraction(1, sidecar.SAMPLE_RATE)
        frame.pts = self.pts
        self.pts += sidecar.SAMPLES_PER_FRAME
        return frame


async def start_sidecar_app() -> tuple[web.AppRunner, str]:
    source = sidecar.PcmSource(None)
    app = sidecar.create_app(source, None)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


async def post_offer(
    session: ClientSession,
    base_url: str,
    call_id: str,
    pc,
) -> dict[str, Any]:
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await sidecar.wait_for_ice_complete(pc)
    assert pc.localDescription is not None

    async with session.post(
        f"{base_url}/offer",
        json={
            "call_id": call_id,
            "type": pc.localDescription.type,
            "sdp": pc.localDescription.sdp,
        },
    ) as response:
        body = await response.json()
        if response.status != 200:
            raise RuntimeError(f"/offer failed ({response.status}): {body}")
        return body


async def drain_decoded_pcm(
    session: ClientSession,
    base_url: str,
    call_id: str,
    *,
    target_bytes: int,
    timeout_s: float,
) -> bytes:
    deadline = asyncio.get_running_loop().time() + timeout_s
    captured = bytearray()
    saw_non_silent = False

    while asyncio.get_running_loop().time() < deadline:
        async with session.get(
            f"{base_url}/calls/{call_id}/audio",
            params={
                "max_bytes": sidecar.DEFAULT_DRAIN_BYTES,
                "wait_ms": 500,
            },
        ) as response:
            body = await response.json()
            if response.status != 200:
                raise RuntimeError(f"audio drain failed ({response.status}): {body}")

        chunk = base64.b64decode(body["pcm_s16le_base64"])
        if body["returned_bytes"] != len(chunk):
            raise RuntimeError("returned_bytes did not match decoded PCM length")
        if chunk:
            captured.extend(chunk)
            if any(chunk):
                saw_non_silent = True
        if saw_non_silent and len(captured) >= target_bytes:
            return bytes(captured)

    if not saw_non_silent:
        raise TimeoutError("sidecar inbound HTTP drain stayed silent")
    raise TimeoutError(
        f"sidecar inbound HTTP drain returned {len(captured)} bytes, "
        f"below target {target_bytes}"
    )


async def send_webrtc_audio_and_drain(
    *,
    pcm: bytes,
    base_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    call_id = "stream-transcribe-loopback"
    pc = sidecar.RTCPeerConnection()
    pc.addTrack(PcmBytesTrack(pcm))
    try:
        async with ClientSession() as session:
            answer = await post_offer(session, base_url, call_id, pc)
            await pc.setRemoteDescription(
                sidecar.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            drained = await drain_decoded_pcm(
                session,
                base_url,
                call_id,
                target_bytes=len(pcm),
                timeout_s=timeout_s,
            )
            return {
                "call_id": call_id,
                "decoded_pcm": drained,
                "audio": sidecar.audio_contract(),
            }
    finally:
        await pc.close()


def run_voice_stream(
    *,
    voice_bin: str,
    output_path: Path,
    text: str,
    voice: str,
    speed: str,
    timeout_s: float,
) -> None:
    command = [
        voice_bin,
        "stream",
        "--quiet",
        "--sample-rate",
        str(sidecar.SAMPLE_RATE),
        "--frame-ms",
        str(sidecar.FRAME_MS),
        "--raw-output",
        str(output_path),
        "--voice",
        voice,
        "--speed",
        speed,
        text,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "voice stream failed"
            + (f": {completed.stderr.strip()[:1000]}" if completed.stderr else "")
        )


def run_voice_stream_transcribe(
    *,
    voice_bin: str,
    input_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            voice_bin,
            "stream-transcribe",
            "--raw-input",
            str(input_path),
            "--sample-rate",
            str(sidecar.SAMPLE_RATE),
            "--frame-ms",
            str(sidecar.FRAME_MS),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "voice stream-transcribe failed"
            + (f": {completed.stderr.strip()[:1000]}" if completed.stderr else "")
        )
    return parse_transcript_event(completed.stdout)


def parse_transcript_event(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "stt.error":
            data = event.get("data")
            raise RuntimeError(f"stream-transcribe returned stt.error: {data}")
        if event.get("event") == "stt.transcribed":
            data = event.get("data")
            if not isinstance(data, dict):
                raise RuntimeError("stt.transcribed event data must be an object")
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("stt.transcribed event did not include text")
            return event
    raise RuntimeError("voice stream-transcribe did not emit stt.transcribed")


def transcript_has_words(transcript: str, expected_words: list[str]) -> bool:
    words = set(re.findall(r"[a-z0-9]+", transcript.lower()))
    return all(word.lower() in words for word in expected_words)


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    workdir = (
        Path(tempfile.mkdtemp(prefix="voice-webrtc-stt-loopback."))
        if args.workdir is None
        else args.workdir.expanduser().resolve()
    )
    workdir.mkdir(parents=True, exist_ok=True)
    remove_workdir = args.workdir is None and not args.keep_workdir

    source_path = workdir / "source.s16le"
    decoded_path = workdir / "decoded.s16le"
    runner, base_url = await start_sidecar_app()
    try:
        run_voice_stream(
            voice_bin=voice_bin,
            output_path=source_path,
            text=args.text,
            voice=args.voice,
            speed=args.speed,
            timeout_s=args.timeout,
        )
        source_pcm = source_path.read_bytes()
        if not source_pcm or not any(source_pcm):
            raise RuntimeError("voice stream produced empty or silent PCM")

        inbound = await send_webrtc_audio_and_drain(
            pcm=source_pcm,
            base_url=base_url,
            timeout_s=args.timeout,
        )
        decoded_pcm = inbound.pop("decoded_pcm")
        decoded_path.write_bytes(decoded_pcm)

        transcript_event = run_voice_stream_transcribe(
            voice_bin=voice_bin,
            input_path=decoded_path,
            timeout_s=args.timeout,
        )
        transcript = transcript_event["data"]["text"]
        if args.expect_word and not transcript_has_words(transcript, args.expect_word):
            raise RuntimeError(
                f"transcript {transcript!r} did not contain expected words "
                f"{args.expect_word!r}"
            )

        retained = not remove_workdir
        return {
            "success": True,
            "sidecar_url": base_url,
            "voice_bin": voice_bin,
            "text": args.text,
            "transcript": transcript,
            "expect_word": args.expect_word,
            "source_pcm_bytes": len(source_pcm),
            "decoded_pcm_bytes": len(decoded_pcm),
            "source_path": str(source_path) if retained else "<temporary>",
            "decoded_path": str(decoded_path) if retained else "<temporary>",
            "retained": retained,
            **inbound,
            "stt": transcript_event["data"],
        }
    finally:
        await runner.cleanup()
        if remove_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--expect-word",
        action="append",
        default=None,
        help="word expected in the transcript; repeatable",
    )
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    if args.expect_word is None:
        args.expect_word = list(DEFAULT_EXPECT_WORDS)
    return args


def main() -> int:
    args = parse_args()
    result = asyncio.run(run_smoke(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
