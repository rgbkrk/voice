from __future__ import annotations

import asyncio
import base64
import importlib.util
from pathlib import Path

import pytest


def load_sidecar():
    path = Path(__file__).with_name("sidecar.py")
    spec = importlib.util.spec_from_file_location("voice_webrtc_sidecar", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit as exc:
        pytest.skip(str(exc))
    return module


def synthetic_pcm_track(sidecar):
    class SyntheticPcmTrack(sidecar.MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self.pts = 0
            samples = [
                (12_000 if index % 2 == 0 else -12_000).to_bytes(
                    2,
                    byteorder="little",
                    signed=True,
                )
                for index in range(sidecar.SAMPLES_PER_FRAME)
            ]
            self.frame_bytes = b"".join(samples)

        async def recv(self):
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

    return SyntheticPcmTrack()


def test_audio_contract_is_voice_pcm_shape():
    sidecar = load_sidecar()

    assert sidecar.audio_contract() == {
        "sample_rate": 48000,
        "channels": 1,
        "frame_ms": 20,
        "encoding": "pcm_s16le",
    }


def test_pcm_source_pads_partial_frame(tmp_path: Path):
    sidecar = load_sidecar()
    source_path = tmp_path / "short.s16le"
    source_path.write_bytes(b"\x01\x02\x03\x04")

    source = sidecar.PcmSource(str(source_path))
    try:
        frame = source.read_frame()
    finally:
        source.close()

    assert len(frame) == sidecar.FRAME_BYTES
    assert frame.startswith(b"\x01\x02\x03\x04")
    assert frame[4:] == b"\x00" * (sidecar.FRAME_BYTES - 4)


def test_call_pcm_source_isolates_per_call_queue():
    sidecar = load_sidecar()

    fallback = sidecar.PcmSource(None)
    call_a = sidecar.CallPcmSource(fallback)
    call_b = sidecar.CallPcmSource(fallback)

    accepted = call_a.write_frame(b"\x01\x00\xff\xff")

    assert accepted == 4
    assert call_a.queued_bytes == 4
    assert call_b.queued_bytes == 0

    frame_b = call_b.read_frame()
    assert frame_b == b"\x00" * sidecar.FRAME_BYTES

    frame_a = call_a.read_frame()
    assert frame_a.startswith(b"\x01\x00\xff\xff")
    assert frame_a[4:] == b"\x00" * (sidecar.FRAME_BYTES - 4)


def test_inbound_pcm_sink_drains_queued_audio():
    sidecar = load_sidecar()
    sink = sidecar.InboundPcmSink(None, max_queue_bytes=6)

    sink.write(b"\x01\x00\x02\x00")
    sink.write(b"\x03\x00\x04\x00")

    assert sink.queued_bytes == 6
    assert sink.drain(2) == b"\x02\x00"
    assert sink.queued_bytes == 4
    assert sink.drain(99) == b"\x03\x00\x04\x00"
    assert sink.queued_bytes == 0


def test_health_reports_audio_contract():
    sidecar = load_sidecar()

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        source = sidecar.PcmSource(None)
        app = sidecar.create_app(source, None)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/health")
            assert response.status == 200
            body = await response.json()
            assert body["ok"] is True
            assert body["sessions"] == 0
            assert body["call_ids"] == []
            assert body["audio"] == sidecar.audio_contract()

    asyncio.run(run())


def test_call_status_and_close_use_session_snapshot():
    sidecar = load_sidecar()

    class FakeSession:
        def __init__(self) -> None:
            self.closed = False

        def snapshot(self):
            return {
                "call_id": "call-1",
                "closed": self.closed,
                "audio": sidecar.audio_contract(),
            }

        async def close(self) -> None:
            self.closed = True

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        source = sidecar.PcmSource(None)
        app = sidecar.create_app(source, None)
        app[sidecar.SESSIONS_KEY]["call-1"] = FakeSession()

        async with TestClient(TestServer(app)) as client:
            status_response = await client.get("/calls/call-1")
            assert status_response.status == 200
            status_body = await status_response.json()
            assert status_body["call_id"] == "call-1"
            assert status_body["closed"] is False
            assert status_body["audio"] == sidecar.audio_contract()

            close_response = await client.post("/calls/call-1/close")
            assert close_response.status == 200
            close_body = await close_response.json()
            assert close_body == {"call_id": "call-1", "closed": True}
            assert app[sidecar.SESSIONS_KEY] == {}

            missing_response = await client.get("/calls/call-1")
            assert missing_response.status == 404
            missing_body = await missing_response.json()
            assert missing_body == {"error": "unknown call_id"}

    asyncio.run(run())


def test_audio_endpoint_queues_pcm_for_call():
    sidecar = load_sidecar()

    class FakeSession:
        def __init__(self, source) -> None:
            self.source = source

        def snapshot(self):
            return {
                "call_id": "call-1",
                "queued_tx_bytes": self.source.queued_bytes,
                "audio": sidecar.audio_contract(),
            }

        async def close(self) -> None:
            pass

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        fallback = sidecar.PcmSource(None)
        source = sidecar.CallPcmSource(fallback)
        app = sidecar.create_app(fallback, None)
        app[sidecar.SESSIONS_KEY]["call-1"] = FakeSession(source)

        payload = {
            "sample_rate": 48000,
            "channels": 1,
            "frame_ms": 20,
            "encoding": "pcm_s16le",
            "pcm_s16le_base64": base64.b64encode(b"\x01\x00\xff\xff").decode("ascii"),
        }
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/calls/call-1/audio", json=payload)
            assert response.status == 200
            body = await response.json()
            assert body["accepted_bytes"] == 4
            assert body["queued_tx_bytes"] == 4

            frame = source.read_frame()
            assert frame.startswith(b"\x01\x00\xff\xff")
            assert frame[4:] == b"\x00" * (sidecar.FRAME_BYTES - 4)

    asyncio.run(run())


def test_audio_endpoint_rejects_mismatched_contract():
    sidecar = load_sidecar()

    class FakeSession:
        def __init__(self, source) -> None:
            self.source = source

        def snapshot(self):
            return {"call_id": "call-1"}

        async def close(self) -> None:
            pass

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        fallback = sidecar.PcmSource(None)
        source = sidecar.CallPcmSource(fallback)
        app = sidecar.create_app(fallback, None)
        app[sidecar.SESSIONS_KEY]["call-1"] = FakeSession(source)

        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/calls/call-1/audio",
                json={
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_ms": 20,
                    "encoding": "pcm_s16le",
                    "pcm_s16le_base64": base64.b64encode(b"\x00\x00").decode("ascii"),
                },
            )
            assert response.status == 400
            body = await response.json()
            assert body == {"error": "sample_rate must be 48000"}
            assert source.queued_bytes == 0

    asyncio.run(run())


def test_audio_endpoint_drains_inbound_pcm_for_call():
    sidecar = load_sidecar()

    class FakeSession:
        def __init__(self) -> None:
            self.inbound = sidecar.InboundPcmSink(None)

        def snapshot(self):
            return {"call_id": "call-1"}

        async def close(self) -> None:
            self.inbound.close()

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        source = sidecar.PcmSource(None)
        session = FakeSession()
        session.inbound.write(b"\x01\x00\xff\xff")
        app = sidecar.create_app(source, None)
        app[sidecar.SESSIONS_KEY]["call-1"] = session

        async with TestClient(TestServer(app)) as client:
            response = await client.get("/calls/call-1/audio?max_bytes=2")
            assert response.status == 200
            body = await response.json()
            assert body["call_id"] == "call-1"
            assert body["returned_bytes"] == 2
            assert body["queued_rx_bytes"] == 2
            assert base64.b64decode(body["pcm_s16le_base64"]) == b"\x01\x00"
            assert body["audio"] == sidecar.audio_contract()

            invalid_response = await client.get("/calls/call-1/audio?max_bytes=0")
            assert invalid_response.status == 400
            invalid_body = await invalid_response.json()
            assert invalid_body == {"error": "max_bytes must be positive"}

            partial_response = await client.get("/calls/call-1/audio?max_bytes=1")
            assert partial_response.status == 400
            partial_body = await partial_response.json()
            assert partial_body == {
                "error": "max_bytes must contain whole s16le samples"
            }

    asyncio.run(run())


def test_offer_loopback_receives_http_queued_audio():
    sidecar = load_sidecar()

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        source = sidecar.PcmSource(None)
        app = sidecar.create_app(source, None)
        pc = sidecar.RTCPeerConnection()
        track_future = asyncio.get_running_loop().create_future()

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio" and not track_future.done():
                track_future.set_result(track)

        pc.addTransceiver("audio", direction="recvonly")

        async with TestClient(TestServer(app)) as client:
            try:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await sidecar.wait_for_ice_complete(pc)
                assert pc.localDescription is not None

                offer_response = await client.post(
                    "/offer",
                    json={
                        "call_id": "call-1",
                        "type": pc.localDescription.type,
                        "sdp": pc.localDescription.sdp,
                    },
                )
                assert offer_response.status == 200
                answer = await offer_response.json()
                assert answer["audio"] == sidecar.audio_contract()

                pcm_frame = (
                    (12_000).to_bytes(2, byteorder="little", signed=True)
                    * sidecar.SAMPLES_PER_FRAME
                )
                audio_response = await client.post(
                    "/calls/call-1/audio",
                    json={
                        "sample_rate": 48000,
                        "channels": 1,
                        "frame_ms": 20,
                        "encoding": "pcm_s16le",
                        "pcm_s16le_base64": base64.b64encode(pcm_frame).decode("ascii"),
                    },
                )
                assert audio_response.status == 200
                audio_body = await audio_response.json()
                assert audio_body["accepted_bytes"] == sidecar.FRAME_BYTES

                await pc.setRemoteDescription(
                    sidecar.RTCSessionDescription(
                        sdp=answer["sdp"],
                        type=answer["type"],
                    )
                )
                track = await asyncio.wait_for(track_future, timeout=5)

                for _ in range(80):
                    frame = await asyncio.wait_for(track.recv(), timeout=2)
                    frame_bytes = b"".join(bytes(plane) for plane in frame.planes)
                    if any(frame_bytes):
                        return
                pytest.fail("remote WebRTC peer only received silence")
            finally:
                await pc.close()

    asyncio.run(run())


def test_offer_loopback_writes_inbound_audio_to_pcm_sink(tmp_path: Path):
    sidecar = load_sidecar()
    rx_path = tmp_path / "inbound.s16le"

    async def wait_for_non_silent_pcm() -> None:
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if rx_path.exists():
                data = rx_path.read_bytes()
                if len(data) >= sidecar.FRAME_BYTES and any(data):
                    return
            await asyncio.sleep(0.05)
        size = rx_path.stat().st_size if rx_path.exists() else 0
        pytest.fail(f"inbound PCM sink stayed silent or empty (bytes={size})")

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        source = sidecar.PcmSource(None)
        app = sidecar.create_app(source, str(rx_path))
        pc = sidecar.RTCPeerConnection()
        pc.addTrack(synthetic_pcm_track(sidecar))

        async with TestClient(TestServer(app)) as client:
            try:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await sidecar.wait_for_ice_complete(pc)
                assert pc.localDescription is not None

                offer_response = await client.post(
                    "/offer",
                    json={
                        "call_id": "call-1",
                        "type": pc.localDescription.type,
                        "sdp": pc.localDescription.sdp,
                    },
                )
                assert offer_response.status == 200
                answer = await offer_response.json()
                await pc.setRemoteDescription(
                    sidecar.RTCSessionDescription(
                        sdp=answer["sdp"],
                        type=answer["type"],
                    )
                )

                await wait_for_non_silent_pcm()
            finally:
                await pc.close()

    asyncio.run(run())


def test_offer_loopback_drains_inbound_audio_over_http():
    sidecar = load_sidecar()

    async def wait_for_non_silent_drain(client) -> None:
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            response = await client.get(
                f"/calls/call-1/audio?max_bytes={sidecar.FRAME_BYTES}"
            )
            assert response.status == 200
            body = await response.json()
            pcm = base64.b64decode(body["pcm_s16le_base64"])
            assert body["returned_bytes"] == len(pcm)
            assert body["audio"] == sidecar.audio_contract()
            if len(pcm) >= sidecar.FRAME_BYTES and any(pcm):
                return
            await asyncio.sleep(0.05)
        pytest.fail("inbound HTTP audio drain stayed silent or empty")

    async def run():
        from aiohttp.test_utils import TestClient, TestServer

        source = sidecar.PcmSource(None)
        app = sidecar.create_app(source, None)
        pc = sidecar.RTCPeerConnection()
        pc.addTrack(synthetic_pcm_track(sidecar))

        async with TestClient(TestServer(app)) as client:
            try:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await sidecar.wait_for_ice_complete(pc)
                assert pc.localDescription is not None

                offer_response = await client.post(
                    "/offer",
                    json={
                        "call_id": "call-1",
                        "type": pc.localDescription.type,
                        "sdp": pc.localDescription.sdp,
                    },
                )
                assert offer_response.status == 200
                answer = await offer_response.json()
                await pc.setRemoteDescription(
                    sidecar.RTCSessionDescription(
                        sdp=answer["sdp"],
                        type=answer["type"],
                    )
                )

                await wait_for_non_silent_drain(client)
            finally:
                await pc.close()

    asyncio.run(run())
