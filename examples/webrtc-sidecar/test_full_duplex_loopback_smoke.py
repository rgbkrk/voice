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
        ],
    )

    args = smoke.parse_args()

    assert args.inbound_text == "testing one two"
    assert args.expect_word == ["testing", "two"]
    assert args.max_queued_tx_ms == 250


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


def test_queued_tx_ms_converts_bytes_to_audio_duration():
    smoke = load_smoke()
    audio = {
        "sample_rate": 48_000,
        "channels": 1,
        "bytes_per_sample": 2,
    }

    assert smoke.queued_tx_ms(1_920, audio) == 20
    assert smoke.queued_tx_ms(96_000, audio) == 1_000


def test_validate_queued_tx_budget_returns_duration_when_within_budget():
    smoke = load_smoke()
    call = {
        "queued_tx_bytes": 1_920,
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
