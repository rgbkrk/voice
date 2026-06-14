from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_smoke():
    path = Path(__file__).with_name("full_duplex_loopback_smoke.py")
    spec = importlib.util.spec_from_file_location(
        "voice_webrtc_full_duplex_loopback_smoke",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_uses_default_expected_words(monkeypatch):
    smoke = load_smoke()
    monkeypatch.setattr(sys, "argv", ["full_duplex_loopback_smoke.py"])

    args = smoke.parse_args()

    assert args.inbound_text == "hello world"
    assert args.outbound_text.startswith("Hello from a full duplex")
    assert args.expect_word == ["hello", "world"]
    assert args.max_queued_tx_ms == smoke.DEFAULT_MAX_QUEUED_TX_MS
    assert args.skip_clear_audio_smoke is False


def test_parse_args_replaces_default_expected_words(monkeypatch):
    smoke = load_smoke()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "full_duplex_loopback_smoke.py",
            "--inbound-text",
            "testing one two",
            "--expect-word",
            "testing",
            "--expect-word",
            "two",
            "--max-queued-tx-ms",
            "250",
            "--skip-clear-audio-smoke",
        ],
    )

    args = smoke.parse_args()

    assert args.inbound_text == "testing one two"
    assert args.expect_word == ["testing", "two"]
    assert args.max_queued_tx_ms == 250
    assert args.skip_clear_audio_smoke is True


def test_parse_args_rejects_negative_queue_budget(monkeypatch):
    smoke = load_smoke()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "full_duplex_loopback_smoke.py",
            "--max-queued-tx-ms",
            "-1",
        ],
    )

    try:
        smoke.parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected negative queue budget to be rejected")


def test_pcm_bytes_to_ms_converts_bytes_to_ceiled_audio_duration():
    smoke = load_smoke()
    audio = {
        "sample_rate": 48_000,
        "channels": 1,
        "bytes_per_sample": 2,
    }

    assert smoke.pcm_bytes_to_ms(0, audio) == 0
    assert smoke.pcm_bytes_to_ms(2, audio) == 1
    assert smoke.pcm_bytes_to_ms(1_920, audio) == 20
    assert smoke.pcm_bytes_to_ms(96_000, audio) == 1_000


def test_queue_ms_from_status_prefers_sidecar_reported_duration():
    smoke = load_smoke()
    audio = {
        "sample_rate": 48_000,
        "channels": 1,
        "bytes_per_sample": 2,
    }
    call = {
        "queued_tx_bytes": 1_920,
        "queued_tx_ms": 21,
    }

    assert smoke.queue_ms_from_status(call, "queued_tx", audio) == 21


def test_queue_ms_from_status_falls_back_to_byte_duration():
    smoke = load_smoke()
    audio = {
        "sample_rate": 48_000,
        "channels": 1,
        "bytes_per_sample": 2,
    }
    call = {
        "queued_rx_bytes": 3_840,
    }

    assert smoke.queue_ms_from_status(call, "queued_rx", audio) == 40


def test_queue_ms_from_status_rejects_invalid_sidecar_duration():
    smoke = load_smoke()

    try:
        smoke.queue_ms_from_status({"queued_tx_ms": "soon"}, "queued_tx", {})
    except RuntimeError as exc:
        assert "queued_tx_ms must be an integer" in str(exc)
    else:
        raise AssertionError("expected invalid sidecar duration to be rejected")


def test_validate_queued_tx_budget_returns_duration_when_within_budget():
    smoke = load_smoke()
    call = {
        "queued_tx_bytes": 1_920,
        "queued_tx_ms": 20,
        "audio": {
            "sample_rate": 48_000,
            "channels": 1,
            "bytes_per_sample": 2,
        },
    }

    assert smoke.validate_queued_tx_budget(call, 20) == 20


def test_validate_queued_tx_budget_rejects_excessive_backlog():
    smoke = load_smoke()
    call = {
        "queued_tx_bytes": 192_000,
        "audio": {
            "sample_rate": 48_000,
            "channels": 1,
            "bytes_per_sample": 2,
        },
    }

    try:
        smoke.validate_queued_tx_budget(call, 1_000)
    except RuntimeError as exc:
        assert "exceeded budget" in str(exc)
        assert "2000 ms > 1000 ms" in str(exc)
    else:
        raise AssertionError("expected excessive queue depth to be rejected")


def test_verify_clear_audio_queues_and_clears_live_call():
    smoke = load_smoke()

    async def run():
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        state = {"queued": 0}
        audio = {
            "sample_rate": 48_000,
            "channels": 1,
            "frame_ms": 20,
            "encoding": "pcm_s16le",
            "frame_bytes": 1_920,
        }

        async def queue_audio(request):
            body = await request.json()
            pcm = smoke.base64.b64decode(body["pcm_s16le_base64"])
            state["queued"] += len(pcm)
            return web.json_response(
                {
                    "queued_tx_bytes": state["queued"],
                    "queued_tx_ms": 200,
                }
            )

        async def clear_audio(_request):
            dropped = state["queued"]
            state["queued"] = 0
            return web.json_response(
                {
                    "dropped_tx_bytes": dropped,
                    "dropped_tx_ms": 200,
                    "queued_tx_bytes": 0,
                    "queued_tx_ms": 0,
                }
            )

        app = web.Application()
        app.router.add_post("/calls/call-1/audio", queue_audio)
        app.router.add_post("/calls/call-1/audio/clear", clear_audio)

        async with TestClient(TestServer(app)) as client:
            return await smoke.verify_clear_audio(
                client.session,
                str(client.make_url("/")).rstrip("/"),
                "call-1",
                audio,
            )

    result = smoke.asyncio.run(run())

    assert result["queued_before_clear_bytes"] > 0
    assert result["dropped_tx_bytes"] == result["queued_before_clear_bytes"]
    assert result["queued_tx_bytes"] == 0


def test_verify_clear_audio_rejects_uncleared_queue():
    smoke = load_smoke()

    async def run():
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        audio = {
            "sample_rate": 48_000,
            "channels": 1,
            "frame_ms": 20,
            "encoding": "pcm_s16le",
            "frame_bytes": 1_920,
        }

        async def queue_audio(request):
            body = await request.json()
            pcm = smoke.base64.b64decode(body["pcm_s16le_base64"])
            return web.json_response(
                {
                    "queued_tx_bytes": len(pcm),
                    "queued_tx_ms": 200,
                }
            )

        async def clear_audio(_request):
            return web.json_response(
                {
                    "dropped_tx_bytes": 0,
                    "queued_tx_bytes": 1_920,
                }
            )

        app = web.Application()
        app.router.add_post("/calls/call-1/audio", queue_audio)
        app.router.add_post("/calls/call-1/audio/clear", clear_audio)

        async with TestClient(TestServer(app)) as client:
            await smoke.verify_clear_audio(
                client.session,
                str(client.make_url("/")).rstrip("/"),
                "call-1",
                audio,
            )

    try:
        smoke.asyncio.run(run())
    except RuntimeError as exc:
        assert "reported no dropped" in str(exc)
    else:
        raise AssertionError("expected uncleared queue to be rejected")
