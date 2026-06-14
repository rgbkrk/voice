#!/usr/bin/env python3
"""Run one full-duplex local WebRTC sidecar turn.

This smoke proves both live-call media directions on the same sidecar session:

    inbound:  voice stream PCM -> local WebRTC sender -> sidecar drain
              -> voice stream-transcribe

    outbound: voice stream via post_voice_stream.py -> sidecar outbound queue
              -> local WebRTC receiver

It does not contact WhatsApp or the Graph API.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from aiohttp import ClientSession, web

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sidecar
from stream_transcribe_loopback_smoke import (
    DEFAULT_EXPECT_WORDS,
    PcmBytesTrack,
    drain_decoded_pcm,
    resolve_executable,
    run_voice_stream,
    run_voice_stream_transcribe,
    transcript_has_words,
)
from voice_stream_loopback_smoke import (
    run_post_voice_stream,
    wait_for_non_silent_track,
)


DEFAULT_INBOUND_TEXT = "hello world"
DEFAULT_OUTBOUND_TEXT = "Hello from a full duplex WebRTC sidecar turn."
DEFAULT_MAX_QUEUED_TX_MS = 1_000
DEFAULT_CLEAR_PROBE_MS = 200


def pcm_bytes_to_ms(byte_count: object, audio: dict[str, Any]) -> int:
    try:
        pcm_bytes = int(byte_count or 0)
        sample_rate = int(audio.get("sample_rate") or 0)
        channels = int(audio.get("channels") or 0)
        bytes_per_sample = int(audio.get("bytes_per_sample") or 0)
    except (TypeError, ValueError):
        return 0

    bytes_per_second = sample_rate * channels * bytes_per_sample
    if pcm_bytes <= 0 or bytes_per_second <= 0:
        return 0
    return (pcm_bytes * 1_000 + bytes_per_second - 1) // bytes_per_second


def queue_ms_from_status(
    call: dict[str, Any],
    field_prefix: str,
    audio: dict[str, Any],
) -> int:
    reported_ms = call.get(f"{field_prefix}_ms")
    if reported_ms is not None:
        try:
            parsed_ms = int(reported_ms)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{field_prefix}_ms must be an integer") from exc
        if parsed_ms < 0:
            raise RuntimeError(f"{field_prefix}_ms must be non-negative")
        return parsed_ms
    return pcm_bytes_to_ms(call.get(f"{field_prefix}_bytes"), audio)


def validate_queued_tx_budget(call: dict[str, Any], max_queued_tx_ms: int) -> int:
    if max_queued_tx_ms < 0:
        raise ValueError("max_queued_tx_ms must be non-negative")
    audio = call.get("audio")
    if not isinstance(audio, dict):
        raise RuntimeError("full-duplex sidecar status did not include audio contract")
    queued_ms = queue_ms_from_status(call, "queued_tx", audio)
    if queued_ms > max_queued_tx_ms:
        raise RuntimeError(
            "sidecar outbound audio queue exceeded budget "
            f"({queued_ms} ms > {max_queued_tx_ms} ms)"
        )
    return queued_ms


async def verify_clear_audio(
    session: ClientSession,
    base_url: str,
    call_id: str,
    audio: dict[str, Any],
    *,
    duration_ms: int = DEFAULT_CLEAR_PROBE_MS,
) -> dict[str, Any]:
    """Queue outbound PCM on a live call, clear it, and verify the queue drains."""

    frame_ms = int(audio.get("frame_ms") or 0)
    frame_bytes = int(audio.get("frame_bytes") or 0)
    if frame_ms <= 0 or frame_bytes <= 0:
        raise RuntimeError("audio contract must include positive frame_ms/frame_bytes")
    frames = max(1, (duration_ms + frame_ms - 1) // frame_ms)
    probe = b"\x01\x00" * ((frame_bytes * frames) // 2)
    payload = {
        "sample_rate": audio["sample_rate"],
        "channels": audio["channels"],
        "frame_ms": audio["frame_ms"],
        "encoding": audio["encoding"],
        "pcm_s16le_base64": base64.b64encode(probe).decode("ascii"),
    }

    async with session.post(f"{base_url}/calls/{call_id}/audio", json=payload) as response:
        queued = await response.json()
        if response.status != 200:
            raise RuntimeError(f"clear probe queue failed ({response.status}): {queued}")

    queued_before_clear = int(queued.get("queued_tx_bytes") or 0)
    if queued_before_clear <= 0:
        raise RuntimeError("clear probe did not leave outbound PCM queued")

    async with session.post(f"{base_url}/calls/{call_id}/audio/clear") as response:
        cleared = await response.json()
        if response.status != 200:
            raise RuntimeError(f"clear audio failed ({response.status}): {cleared}")

    dropped = int(cleared.get("dropped_tx_bytes") or 0)
    remaining = int(cleared.get("queued_tx_bytes") or 0)
    if dropped <= 0:
        raise RuntimeError("clear audio reported no dropped outbound PCM")
    if remaining != 0:
        raise RuntimeError(f"clear audio left {remaining} queued outbound bytes")

    return {
        "queued_before_clear_bytes": queued_before_clear,
        "queued_before_clear_ms": queued.get("queued_tx_ms"),
        "dropped_tx_bytes": dropped,
        "dropped_tx_ms": cleared.get("dropped_tx_ms"),
        "queued_tx_bytes": remaining,
        "queued_tx_ms": cleared.get("queued_tx_ms"),
    }


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


async def verify_full_duplex_call(
    *,
    args: argparse.Namespace,
    base_url: str,
    inbound_pcm: bytes,
) -> dict[str, Any]:
    call_id = "full-duplex-loopback"
    pc = sidecar.RTCPeerConnection()
    track_future = asyncio.get_running_loop().create_future()

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio" and not track_future.done():
            track_future.set_result(track)

    pc.addTrack(PcmBytesTrack(inbound_pcm))
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
                    text=args.outbound_text,
                    voice=args.voice,
                    speed=args.speed,
                    timeout_s=args.timeout,
                )
            )
            drain_task = asyncio.create_task(
                drain_decoded_pcm(
                    session,
                    base_url,
                    call_id,
                    target_bytes=len(inbound_pcm),
                    timeout_s=args.timeout,
                )
            )
            try:
                outbound_webrtc_bytes = await wait_for_non_silent_track(
                    track,
                    timeout_s=args.timeout,
                )
                decoded_pcm = await drain_task
                return_code, stderr = await post_task
            except Exception:
                for task in (post_task, drain_task):
                    task.cancel()
                await asyncio.gather(post_task, drain_task, return_exceptions=True)
                raise

            if return_code != 0:
                raise RuntimeError(
                    f"post_voice_stream.py exited with {return_code}: {stderr.strip()}"
                )

            async with session.get(f"{base_url}/calls/{call_id}") as response:
                status = await response.json()
                if response.status != 200:
                    raise RuntimeError(f"status failed ({response.status}): {status}")

            clear_audio = None
            if not args.skip_clear_audio_smoke:
                clear_audio = await verify_clear_audio(
                    session,
                    base_url,
                    call_id,
                    status["audio"],
                )

            return {
                "call_id": call_id,
                "decoded_pcm": decoded_pcm,
                "outbound_webrtc_bytes": outbound_webrtc_bytes,
                "queued_tx_bytes": status.get("queued_tx_bytes"),
                "queued_tx_ms": status.get("queued_tx_ms"),
                "queued_rx_bytes": status.get("queued_rx_bytes"),
                "queued_rx_ms": status.get("queued_rx_ms"),
                "max_tx_queue_ms": status.get("max_tx_queue_ms"),
                "max_rx_queue_bytes": status.get("max_rx_queue_bytes"),
                "max_rx_queue_ms": status.get("max_rx_queue_ms"),
                "audio": sidecar.audio_contract(),
                "clear_audio": clear_audio if clear_audio is not None else "skipped",
            }
    finally:
        await pc.close()


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    args.voice_bin = voice_bin
    workdir = (
        Path(tempfile.mkdtemp(prefix="voice-webrtc-full-duplex."))
        if args.workdir is None
        else args.workdir.expanduser().resolve()
    )
    workdir.mkdir(parents=True, exist_ok=True)
    remove_workdir = args.workdir is None and not args.keep_workdir

    inbound_path = workdir / "inbound-source.s16le"
    decoded_path = workdir / "sidecar-decoded.s16le"
    runner, base_url = await start_sidecar_app()
    try:
        run_voice_stream(
            voice_bin=voice_bin,
            output_path=inbound_path,
            text=args.inbound_text,
            voice=args.voice,
            speed=args.speed,
            timeout_s=args.timeout,
        )
        inbound_pcm = inbound_path.read_bytes()
        if not inbound_pcm or not any(inbound_pcm):
            raise RuntimeError("voice stream produced empty or silent inbound PCM")

        call = await verify_full_duplex_call(
            args=args,
            base_url=base_url,
            inbound_pcm=inbound_pcm,
        )
        decoded_pcm = call.pop("decoded_pcm")
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
        queued_tx_ms_value = validate_queued_tx_budget(
            call,
            args.max_queued_tx_ms,
        )
        audio = call["audio"]
        queued_rx_ms_value = queue_ms_from_status(call, "queued_rx", audio)

        retained = not remove_workdir
        return {
            "success": True,
            "sidecar_url": base_url,
            "voice_bin": voice_bin,
            "inbound_text": args.inbound_text,
            "outbound_text": args.outbound_text,
            "transcript": transcript,
            "expect_word": args.expect_word,
            "source_pcm_bytes": len(inbound_pcm),
            "decoded_pcm_bytes": len(decoded_pcm),
            "inbound_path": str(inbound_path) if retained else "<temporary>",
            "decoded_path": str(decoded_path) if retained else "<temporary>",
            "retained": retained,
            "max_queued_tx_ms": args.max_queued_tx_ms,
            **call,
            "queued_tx_ms": queued_tx_ms_value,
            "queued_rx_ms": queued_rx_ms_value,
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
    parser.add_argument("--inbound-text", default=DEFAULT_INBOUND_TEXT)
    parser.add_argument("--outbound-text", default=DEFAULT_OUTBOUND_TEXT)
    parser.add_argument(
        "--expect-word",
        action="append",
        default=None,
        help="word expected in the inbound transcript; repeatable",
    )
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--max-queued-tx-ms",
        type=int,
        default=DEFAULT_MAX_QUEUED_TX_MS,
        help=(
            "Maximum outbound sidecar queue depth allowed at the end of the "
            "smoke. Set to 0 to require a fully drained queue."
        ),
    )
    parser.add_argument(
        "--skip-clear-audio-smoke",
        action="store_true",
        help="skip live-call /audio/clear verification after the media turn",
    )
    args = parser.parse_args()
    if args.max_queued_tx_ms < 0:
        parser.error("--max-queued-tx-ms must be non-negative")
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
