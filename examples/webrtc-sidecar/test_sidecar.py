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

        source = sidecar.PcmSource(None)
        app = sidecar.create_app(source, None)
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

        source = sidecar.PcmSource(None)
        app = sidecar.create_app(source, None)
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
