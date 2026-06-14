from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load_smoke():
    path = Path(__file__).with_name("stream_transcribe_loopback_smoke.py")
    spec = importlib.util.spec_from_file_location(
        "voice_webrtc_stream_transcribe_loopback_smoke",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iter_pcm_frames_pads_final_frame():
    smoke = load_smoke()

    frames = list(smoke.iter_pcm_frames(b"abcde", frame_bytes=2))

    assert frames == [b"ab", b"cd", b"e\x00"]


def test_iter_pcm_frames_rejects_invalid_frame_size():
    smoke = load_smoke()

    with pytest.raises(ValueError, match="positive"):
        list(smoke.iter_pcm_frames(b"abc", frame_bytes=0))


def test_parse_transcript_event_ignores_non_json_lines():
    smoke = load_smoke()
    event = {
        "event": "stt.transcribed",
        "data": {"text": "Hello World", "frames": 3},
    }

    parsed = smoke.parse_transcript_event(
        "Streaming: ignored progress\n" + json.dumps(event) + "\n"
    )

    assert parsed == event


def test_parse_transcript_event_rejects_stt_error():
    smoke = load_smoke()

    with pytest.raises(RuntimeError, match="stt.error"):
        smoke.parse_transcript_event(
            json.dumps({"event": "stt.error", "data": {"error": "boom"}})
        )


def test_parse_transcript_event_requires_text():
    smoke = load_smoke()

    with pytest.raises(RuntimeError, match="did not include text"):
        smoke.parse_transcript_event(
            json.dumps({"event": "stt.transcribed", "data": {"text": " "}})
        )


def test_transcript_has_expected_words_case_insensitive():
    smoke = load_smoke()

    assert smoke.transcript_has_words("Hello, world!", ["hello", "world"])
    assert not smoke.transcript_has_words("Hello there", ["hello", "world"])


def test_parse_args_uses_default_expected_words(monkeypatch):
    smoke = load_smoke()
    monkeypatch.setattr(sys, "argv", ["stream_transcribe_loopback_smoke.py"])

    args = smoke.parse_args()

    assert args.expect_word == ["hello", "world"]


def test_parse_args_replaces_default_expected_words(monkeypatch):
    smoke = load_smoke()
    monkeypatch.setattr(
        sys,
        "argv",
        ["stream_transcribe_loopback_smoke.py", "--expect-word", "testing"],
    )

    args = smoke.parse_args()

    assert args.expect_word == ["testing"]
