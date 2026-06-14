#!/usr/bin/env python3
"""Run an in-process WebRTC sidecar loopback smoke test.

The smoke creates a local HTTP sidecar app, performs an SDP offer/answer with a
local aiortc peer, proves queued HTTP PCM reaches a WebRTC audio track, and
proves inbound WebRTC audio can be drained back over the sidecar HTTP API.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from typing import Any

from aiohttp import ClientSession, web

import sidecar


def pcm_frame(sample: int = 12_000) -> bytes:
    return (
        sample.to_bytes(2, byteorder="little", signed=True)
        * sidecar.SAMPLES_PER_FRAME
    )


def audio_payload(frame: bytes) -> dict[str, object]:
    return {
        "sample_rate": sidecar.SAMPLE_RATE,
        "channels": sidecar.CHANNELS,
        "frame_ms": sidecar.FRAME_MS,
        "encoding": sidecar.AUDIO_CONTRACT["encoding"],
        "pcm_s16le_base64": base64.b64encode(frame).decode("ascii"),
    }


class SyntheticPcmTrack(sidecar.MediaStreamTrack):
    """Local non-silent audio source for the inbound sidecar path."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.pts = 0
        self.frame_bytes = pcm_frame(-12_000)

    async def recv(self) -> sidecar.av.AudioFrame:
        await asyncio.sleep(sidecar.FRAME_MS / 1_000)
        frame = sidecar.av.AudioFrame(
            format="s16",
            layout="mono",
            samples=sidecar.SAMPLES_PER_FRAME,
        )
        frame.planes[0].update(self.frame_bytes)
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


async def wait_for_non_silent_track(track, *, timeout_s: float) -> int:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        frame = await asyncio.wait_for(track.recv(), timeout=2)
        frame_bytes = b"".join(bytes(plane) for plane in frame.planes)
        if any(frame_bytes):
            return len(frame_bytes)
    raise TimeoutError("remote WebRTC peer only received silence")


async def wait_for_non_silent_drain(
    session: ClientSession,
    base_url: str,
    call_id: str,
    *,
    timeout_s: float,
) -> int:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        async with session.get(
            f"{base_url}/calls/{call_id}/audio",
            params={"max_bytes": sidecar.FRAME_BYTES, "wait_ms": 500},
        ) as response:
            body = await response.json()
            if response.status != 200:
                raise RuntimeError(f"audio drain failed ({response.status}): {body}")
        pcm = base64.b64decode(body["pcm_s16le_base64"])
        if body["returned_bytes"] != len(pcm):
            raise RuntimeError("returned_bytes did not match decoded PCM length")
        if len(pcm) >= sidecar.FRAME_BYTES and any(pcm):
            return len(pcm)
    raise TimeoutError("sidecar inbound HTTP drain stayed silent")


async def verify_outbound(session: ClientSession, base_url: str, timeout_s: float) -> int:
    pc = sidecar.RTCPeerConnection()
    track_future = asyncio.get_running_loop().create_future()

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio" and not track_future.done():
            track_future.set_result(track)

    pc.addTransceiver("audio", direction="recvonly")
    try:
        answer = await post_offer(session, base_url, "loopback-outbound", pc)
        async with session.post(
            f"{base_url}/calls/loopback-outbound/audio",
            json=audio_payload(pcm_frame()),
        ) as response:
            body = await response.json()
            if response.status != 200:
                raise RuntimeError(f"audio post failed ({response.status}): {body}")
            if body.get("accepted_bytes") != sidecar.FRAME_BYTES:
                raise RuntimeError(f"unexpected accepted_bytes: {body}")

        await pc.setRemoteDescription(
            sidecar.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        track = await asyncio.wait_for(track_future, timeout=timeout_s)
        return await wait_for_non_silent_track(track, timeout_s=timeout_s)
    finally:
        await pc.close()


async def verify_inbound(session: ClientSession, base_url: str, timeout_s: float) -> int:
    pc = sidecar.RTCPeerConnection()
    pc.addTrack(SyntheticPcmTrack())
    try:
        answer = await post_offer(session, base_url, "loopback-inbound", pc)
        await pc.setRemoteDescription(
            sidecar.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        return await wait_for_non_silent_drain(
            session,
            base_url,
            "loopback-inbound",
            timeout_s=timeout_s,
        )
    finally:
        await pc.close()


async def run_smoke(timeout_s: float) -> dict[str, Any]:
    runner, base_url = await start_sidecar_app()
    try:
        async with ClientSession() as session:
            outbound_bytes = await verify_outbound(session, base_url, timeout_s)
            inbound_bytes = await verify_inbound(session, base_url, timeout_s)
            return {
                "success": True,
                "sidecar_url": base_url,
                "outbound_webrtc_bytes": outbound_bytes,
                "inbound_drain_bytes": inbound_bytes,
                "audio": sidecar.audio_contract(),
            }
    finally:
        await runner.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run_smoke(args.timeout))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
