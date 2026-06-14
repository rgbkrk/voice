#!/usr/bin/env python3
"""Run `voice stream` through the WebRTC sidecar loopback.

This smoke proves the real Rust streaming command can feed the Python sidecar's
outbound audio queue and arrive at a local WebRTC peer as non-silent audio. It
does not contact WhatsApp or the Graph API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from aiohttp import ClientSession, web

import sidecar


DEFAULT_TEXT = "Hello from voice stream through the WebRTC sidecar."


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


async def wait_for_non_silent_track(track, *, timeout_s: float) -> int:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        frame = await asyncio.wait_for(track.recv(), timeout=2)
        frame_bytes = b"".join(bytes(plane) for plane in frame.planes)
        if any(frame_bytes):
            return len(frame_bytes)
    raise TimeoutError("remote WebRTC peer only received silence")


def post_voice_stream_path() -> Path:
    return Path(__file__).with_name("post_voice_stream.py")


async def run_post_voice_stream(
    *,
    call_id: str,
    sidecar_url: str,
    voice_bin: str,
    text: str,
    voice: str,
    speed: str,
    timeout_s: float,
) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(post_voice_stream_path()),
        call_id,
        text,
        "--sidecar-url",
        sidecar_url,
        "--voice-bin",
        voice_bin,
        "--voice",
        voice,
        "--speed",
        speed,
        "--timeout",
        str(timeout_s),
        "--quiet",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        _, stderr = await process.communicate()
        raise TimeoutError("post_voice_stream.py timed out")
    except asyncio.CancelledError:
        process.kill()
        await process.communicate()
        raise
    return int(process.returncode or 0), stderr.decode("utf-8", errors="replace")


async def verify_voice_stream_outbound(args: argparse.Namespace, base_url: str) -> dict[str, Any]:
    call_id = "voice-stream-loopback"
    pc = sidecar.RTCPeerConnection()
    track_future = asyncio.get_running_loop().create_future()

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio" and not track_future.done():
            track_future.set_result(track)

    pc.addTransceiver("audio", direction="recvonly")
    try:
        async with ClientSession() as session:
            answer = await post_offer(session, base_url, call_id, pc)
            await pc.setRemoteDescription(
                sidecar.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            track = await asyncio.wait_for(track_future, timeout=args.timeout)

            post_task = asyncio.create_task(
                run_post_voice_stream(
                    call_id=call_id,
                    sidecar_url=base_url,
                    voice_bin=args.voice_bin,
                    text=args.text,
                    voice=args.voice,
                    speed=args.speed,
                    timeout_s=args.timeout,
                )
            )
            try:
                webrtc_bytes = await wait_for_non_silent_track(
                    track,
                    timeout_s=args.timeout,
                )
                return_code, stderr = await post_task
            except Exception:
                post_task.cancel()
                try:
                    await post_task
                except asyncio.CancelledError:
                    pass
                raise

            if return_code != 0:
                raise RuntimeError(
                    f"post_voice_stream.py exited with {return_code}: {stderr.strip()}"
                )

            status_url = f"{base_url}/calls/{call_id}"
            async with session.get(status_url) as response:
                status = await response.json()
                if response.status != 200:
                    raise RuntimeError(f"status failed ({response.status}): {status}")

            return {
                "call_id": call_id,
                "outbound_webrtc_bytes": webrtc_bytes,
                "queued_tx_bytes": status.get("queued_tx_bytes"),
                "audio": sidecar.audio_contract(),
            }
    finally:
        await pc.close()


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    runner, base_url = await start_sidecar_app()
    try:
        outbound = await verify_voice_stream_outbound(args, base_url)
        return {
            "success": True,
            "sidecar_url": base_url,
            "voice_bin": args.voice_bin,
            "text": args.text,
            **outbound,
        }
    finally:
        await runner.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=os.environ.get("VOICE_BIN", "voice"))
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run_smoke(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
