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
        ],
    )

    args = smoke.parse_args()

    assert args.inbound_text == "testing one two"
    assert args.expect_word == ["testing", "two"]
