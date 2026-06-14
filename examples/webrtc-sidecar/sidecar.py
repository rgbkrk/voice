#!/usr/bin/env python3
"""Minimal local WebRTC sidecar for the voice PCM stream contract.

This example is intentionally small:

- local HTTP accepts a remote SDP offer and returns an SDP answer
- outbound audio is raw pcm_s16le, mono, 48 kHz, 20 ms frames
- inbound WebRTC audio is decoded to raw pcm_s16le, mono, 48 kHz

The WhatsApp Graph API and Hermes session loop stay outside this process.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from fractions import Fraction
import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any

try:
    import av
    from aiohttp import web
    from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError
except ImportError as exc:  # pragma: no cover - depends on optional extras
    raise SystemExit(
        "Missing WebRTC sidecar dependencies. Install with:\n"
        "  python -m pip install -r examples/webrtc-sidecar/requirements.txt"
    ) from exc


LOGGER = logging.getLogger("voice-webrtc-sidecar")
SAMPLE_RATE = 48_000
FRAME_MS = 20
CHANNELS = 1
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1_000
BYTES_PER_SAMPLE = 2
FRAME_BYTES = SAMPLES_PER_FRAME * CHANNELS * BYTES_PER_SAMPLE


class PcmSource:
    """Non-blocking raw PCM source with silence fallback."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.fd: int | None = None
        self.buffer = bytearray()
        if not path:
            return

        if path == "-":
            self.fd = os.dup(sys.stdin.fileno())
        else:
            self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        os.set_blocking(self.fd, False)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    @property
    def queued_bytes(self) -> int:
        return len(self.buffer)

    def write_frame(self, pcm_s16le: bytes) -> int:
        self.buffer.extend(pcm_s16le)
        return len(pcm_s16le)

    def read_frame(self) -> bytes:
        if self.fd is not None:
            while len(self.buffer) < FRAME_BYTES:
                try:
                    chunk = os.read(self.fd, FRAME_BYTES - len(self.buffer))
                except BlockingIOError:
                    break
                if not chunk:
                    break
                self.buffer.extend(chunk)

        if len(self.buffer) >= FRAME_BYTES:
            frame = bytes(self.buffer[:FRAME_BYTES])
            del self.buffer[:FRAME_BYTES]
            return frame

        frame = bytes(self.buffer)
        self.buffer.clear()
        return frame + (b"\x00" * (FRAME_BYTES - len(frame)))


class CallPcmSource:
    """Per-call PCM queue with a process-level source as fallback."""

    def __init__(self, fallback: PcmSource) -> None:
        self.fallback = fallback
        self.buffer = bytearray()

    @property
    def queued_bytes(self) -> int:
        return len(self.buffer)

    def write_frame(self, pcm_s16le: bytes) -> int:
        self.buffer.extend(pcm_s16le)
        return len(pcm_s16le)

    def read_frame(self) -> bytes:
        if len(self.buffer) >= FRAME_BYTES:
            frame = bytes(self.buffer[:FRAME_BYTES])
            del self.buffer[:FRAME_BYTES]
            return frame

        if self.buffer:
            frame = bytes(self.buffer)
            self.buffer.clear()
            return frame + (b"\x00" * (FRAME_BYTES - len(frame)))

        return self.fallback.read_frame()


class VoicePcmAudioTrack(MediaStreamTrack):
    """Outbound WebRTC audio track backed by local pcm_s16le frames."""

    kind = "audio"

    def __init__(self, source: PcmSource | CallPcmSource) -> None:
        super().__init__()
        self.source = source
        self.pts = 0
        self.started_at: float | None = None
        self.next_frame_at: float | None = None

    async def recv(self) -> av.AudioFrame:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self.started_at is None:
            self.started_at = now
            self.next_frame_at = now
        elif self.next_frame_at is not None and now < self.next_frame_at:
            await asyncio.sleep(self.next_frame_at - now)

        assert self.next_frame_at is not None
        self.next_frame_at += FRAME_MS / 1_000

        frame = av.AudioFrame(format="s16", layout="mono", samples=SAMPLES_PER_FRAME)
        frame.planes[0].update(self.source.read_frame())
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = Fraction(1, SAMPLE_RATE)
        frame.pts = self.pts
        self.pts += SAMPLES_PER_FRAME
        return frame


async def write_inbound_pcm(track: MediaStreamTrack, path: str | None) -> None:
    if not path:
        while True:
            try:
                await track.recv()
            except MediaStreamError:
                return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

    with target.open("ab") as sink:
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            for out in resampler.resample(frame):
                sink.write(bytes(out.planes[0]))
                sink.flush()


class CallSession:
    def __init__(self, call_id: str, source: PcmSource, rx_pcm: str | None) -> None:
        self.call_id = call_id
        self.source = CallPcmSource(source)
        self.rx_pcm = rx_pcm
        self.pc = RTCPeerConnection()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.closed = False
        self.pc.addTrack(VoicePcmAudioTrack(self.source))

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            LOGGER.info("call %s received %s track", self.call_id, track.kind)
            if track.kind == "audio":
                task = asyncio.create_task(write_inbound_pcm(track, self.rx_pcm))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            LOGGER.info("call %s connection state: %s", self.call_id, self.pc.connectionState)
            if self.pc.connectionState == "failed":
                await self.close()

    async def answer(self, remote_sdp: str, remote_type: str) -> RTCSessionDescription:
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=remote_sdp, type=remote_type)
        )
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await wait_for_ice_complete(self.pc)
        assert self.pc.localDescription is not None
        return self.pc.localDescription

    def snapshot(self) -> dict[str, Any]:
        """Return call state useful for local health checks and Hermes debugging."""
        return {
            "call_id": self.call_id,
            "closed": self.closed,
            "connection_state": self.pc.connectionState,
            "ice_connection_state": self.pc.iceConnectionState,
            "ice_gathering_state": self.pc.iceGatheringState,
            "signaling_state": self.pc.signalingState,
            "tasks": len(self.tasks),
            "queued_tx_bytes": self.source.queued_bytes,
            "audio": audio_contract(),
        }

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.pc.close()


SESSIONS_KEY = web.AppKey("sessions", dict[str, CallSession])


async def wait_for_ice_complete(pc: RTCPeerConnection, timeout: float = 5.0) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except TimeoutError:
        LOGGER.warning("ICE gathering did not complete within %.1fs", timeout)


def json_error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"error": message}, status=status)


def audio_contract() -> dict[str, Any]:
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "frame_ms": FRAME_MS,
        "encoding": "pcm_s16le",
    }


def required_int(body: dict[str, Any], key: str, default: int | None = None) -> int:
    value = body.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def decode_pcm_payload(body: dict[str, Any]) -> bytes:
    sample_rate = required_int(body, "sample_rate")
    channels = required_int(body, "channels", CHANNELS)
    frame_ms = required_int(body, "frame_ms", FRAME_MS)
    encoding = str(body.get("encoding") or "").strip().lower()
    payload = str(body.get("pcm_s16le_base64") or "").strip()

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"sample_rate must be {SAMPLE_RATE}")
    if channels != CHANNELS:
        raise ValueError(f"channels must be {CHANNELS}")
    if frame_ms != FRAME_MS:
        raise ValueError(f"frame_ms must be {FRAME_MS}")
    if encoding not in {"pcm_s16le", "pcm_s16_le"}:
        raise ValueError("encoding must be pcm_s16le")
    if not payload:
        raise ValueError("pcm_s16le_base64 is required")

    try:
        pcm = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("pcm_s16le_base64 is not valid base64") from exc
    if not pcm:
        raise ValueError("decoded PCM payload is empty")
    if len(pcm) % BYTES_PER_SAMPLE != 0:
        raise ValueError("decoded PCM payload must contain whole s16le samples")
    return pcm


def create_app(source: PcmSource, rx_pcm: str | None) -> web.Application:
    app = web.Application()
    sessions: dict[str, CallSession] = {}
    app[SESSIONS_KEY] = sessions

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "sessions": len(sessions),
                "call_ids": sorted(sessions.keys()),
                "audio": audio_contract(),
            }
        )

    async def offer(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return json_error("request body must be JSON")

        call_id = str(body.get("call_id") or "").strip()
        remote_sdp = str(body.get("sdp") or body.get("remote_sdp") or "").strip()
        remote_type = str(body.get("type") or "offer").strip()

        if not call_id:
            return json_error("call_id is required")
        if not remote_sdp:
            return json_error("sdp is required")
        if remote_type != "offer":
            return json_error("only SDP offers are supported")
        if call_id in sessions:
            await sessions.pop(call_id).close()

        session = CallSession(call_id, source, rx_pcm)
        sessions[call_id] = session
        try:
            answer = await session.answer(remote_sdp, remote_type)
        except Exception:
            await sessions.pop(call_id, session).close()
            LOGGER.exception("call %s failed to create answer", call_id)
            return json_error("failed to create SDP answer", status=500)

        return web.json_response(
            {
                "call_id": call_id,
                "type": answer.type,
                "sdp": answer.sdp,
                "audio": audio_contract(),
                "state": session.snapshot(),
            }
        )

    async def call_status(request: web.Request) -> web.Response:
        call_id = request.match_info["call_id"]
        session = sessions.get(call_id)
        if session is None:
            return json_error("unknown call_id", status=404)
        return web.json_response(session.snapshot())

    async def send_audio(request: web.Request) -> web.Response:
        call_id = request.match_info["call_id"]
        session = sessions.get(call_id)
        if session is None:
            return json_error("unknown call_id", status=404)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return json_error("request body must be JSON")
        try:
            pcm = decode_pcm_payload(body)
        except ValueError as exc:
            return json_error(str(exc))

        accepted_bytes = session.source.write_frame(pcm)
        return web.json_response(
            {
                "call_id": call_id,
                "accepted_bytes": accepted_bytes,
                "queued_tx_bytes": session.source.queued_bytes,
                "audio": audio_contract(),
            }
        )

    async def close_call(request: web.Request) -> web.Response:
        call_id = request.match_info["call_id"]
        session = sessions.pop(call_id, None)
        if session is None:
            return json_error("unknown call_id", status=404)
        await session.close()
        return web.json_response({"call_id": call_id, "closed": True})

    async def cleanup(_: web.Application) -> None:
        for session in list(sessions.values()):
            await session.close()
        source.close()

    app.router.add_get("/health", health)
    app.router.add_post("/offer", offer)
    app.router.add_get("/calls/{call_id}", call_status)
    app.router.add_post("/calls/{call_id}/audio", send_audio)
    app.router.add_post("/calls/{call_id}/close", close_call)
    app.on_cleanup.append(cleanup)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--tx-pcm",
        help="raw pcm_s16le mono 48 kHz source path, FIFO, or '-' for stdin",
    )
    parser.add_argument(
        "--rx-pcm",
        help="raw pcm_s16le mono 48 kHz sink for decoded inbound WebRTC audio",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source = PcmSource(args.tx_pcm)
    app = create_app(source, args.rx_pcm)
    runner = web.AppRunner(app)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop = asyncio.Event()
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is not None:
            loop.add_signal_handler(signum, stop.set)

    async def run() -> None:
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        await site.start()
        LOGGER.info("listening on http://%s:%d", args.host, args.port)
        await stop.wait()
        await runner.cleanup()

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
