from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import threading


def load_bridge():
    path = Path(__file__).with_name("post_voice_stream.py")
    spec = importlib.util.spec_from_file_location("voice_webrtc_post_voice_stream", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_audio_contract_matches_sidecar_contract():
    bridge = load_bridge()

    contract = bridge.load_audio_contract()

    assert contract.sample_rate == 48_000
    assert contract.channels == 1
    assert contract.frame_ms == 20
    assert contract.encoding == "pcm_s16le"
    assert contract.frame_bytes == 1_920
    assert contract.default_drain_bytes == 96_000
    assert contract.max_outbound_queue_bytes == 960_000
    assert contract.max_drain_wait_ms == 5_000


def test_load_audio_contract_falls_back_to_voice_stream_contract(monkeypatch, tmp_path: Path):
    bridge = load_bridge()
    calls = []
    contract_json = {
        "audio": {
            "sample_rate": 48_000,
            "channels": 1,
            "frame_ms": 20,
            "encoding": "pcm_s16le",
            "frame_bytes": 1_920,
            "default_drain_bytes": 96_000,
            "max_outbound_queue_bytes": 960_000,
            "max_drain_wait_ms": 5_000,
        }
    }

    class Completed:
        stdout = json.dumps(contract_json)

    def fake_run(command, *, capture_output, text, timeout, check):
        calls.append((command, capture_output, text, timeout, check))
        return Completed()

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)

    contract = bridge.load_audio_contract(
        tmp_path / "missing.json",
        voice_bin="/opt/voice",
    )

    assert contract.sample_rate == 48_000
    assert contract.frame_bytes == 1_920
    assert calls == [(["/opt/voice", "stream-contract"], True, True, 5, True)]


def test_load_audio_contract_rejects_invalid_audio_shape(tmp_path: Path):
    bridge = load_bridge()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "audio": {
                    "sample_rate": 48_000,
                    "channels": 2,
                    "frame_ms": 20,
                    "encoding": "pcm_s16le",
                    "frame_bytes": 3_840,
                    "default_drain_bytes": 3_840,
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        bridge.load_audio_contract(contract_path)
    except ValueError as exc:
        assert "mono" in str(exc)
    else:
        raise AssertionError("expected invalid sidecar contract to be rejected")


def test_sidecar_audio_url_escapes_call_id():
    bridge = load_bridge()

    url = bridge.sidecar_audio_url("http://127.0.0.1:8787/", "wamid/call 1")

    assert url == "http://127.0.0.1:8787/calls/wamid%2Fcall%201/audio"


def test_iter_pcm_frames_reads_exact_frames_and_pads_final():
    bridge = load_bridge()
    stream = BytesIO(b"a" * 5 + b"bb")

    frames = list(bridge.iter_pcm_frames(stream, 5))

    assert frames == [b"aaaaa", b"bb\x00\x00\x00"]


def test_build_audio_payload_uses_contract_shape():
    bridge = load_bridge()
    contract = bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
    )

    payload = bridge.build_audio_payload(contract, b"\x01\x00\xff\xff")

    assert payload == {
        "sample_rate": 48_000,
        "channels": 1,
        "frame_ms": 20,
        "encoding": "pcm_s16le",
        "pcm_s16le_base64": base64.b64encode(b"\x01\x00\xff\xff").decode("ascii"),
    }


def test_build_audio_payload_rejects_partial_sample():
    bridge = load_bridge()
    contract = bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
    )

    try:
        bridge.build_audio_payload(contract, b"\x01")
    except ValueError as exc:
        assert "whole s16le samples" in str(exc)
    else:
        raise AssertionError("expected partial s16le sample to be rejected")


def test_post_audio_frame_sends_json_payload_to_sidecar_endpoint():
    bridge = load_bridge()
    contract = bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
    )
    received: list[tuple[str, dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers["content-length"])
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"accepted_bytes":2}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/calls/call-1/audio"

        response = bridge.post_audio_frame(url, contract, b"\x01\x00", 1.0)

        assert response == {"accepted_bytes": 2}
        assert received == [
            (
                "/calls/call-1/audio",
                {
                    "sample_rate": 48_000,
                    "channels": 1,
                    "frame_ms": 20,
                    "encoding": "pcm_s16le",
                    "pcm_s16le_base64": "AQA=",
                },
            )
        ]
    finally:
        server.shutdown()
        thread.join(timeout=1)


def test_post_audio_frame_marks_429_as_retryable_backpressure():
    bridge = load_bridge()
    contract = bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(429)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"outbound PCM queue is full"}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/calls/call-1/audio"

        try:
            bridge.post_audio_frame(url, contract, b"\x01\x00", 1.0)
        except bridge.SidecarAudioPostError as exc:
            assert exc.status_code == 429
            assert exc.retryable is True
            assert "outbound PCM queue is full" in exc.body
        else:
            raise AssertionError("expected 429 to raise a typed sidecar error")
    finally:
        server.shutdown()
        thread.join(timeout=1)


def test_stop_voice_process_terminates_running_child():
    bridge = load_bridge()

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = FakeProcess()

    return_code = bridge.stop_voice_process(process)

    assert return_code == -15
    assert process.terminated is True
    assert process.killed is False


def test_build_voice_stream_command_uses_contract_and_handoff_flags():
    bridge = load_bridge()
    contract = bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
    )
    args = argparse.Namespace(
        voice_bin="/usr/local/bin/voice",
        voice="af_heart",
        speed="1.1",
        markdown=True,
        sub=["Hermes=her-meez"],
        sub_file="/tmp/subs",
        input_file=None,
        text=["Hello", "from", "WebRTC"],
    )

    command = bridge.build_voice_stream_command(args, contract)

    assert command == [
        "/usr/local/bin/voice",
        "stream",
        "--sample-rate",
        "48000",
        "--frame-ms",
        "20",
        "--raw-output",
        "-",
        "--voice",
        "af_heart",
        "--speed",
        "1.1",
        "--markdown",
        "--sub",
        "Hermes=her-meez",
        "--sub-file",
        "/tmp/subs",
        "Hello",
        "from",
        "WebRTC",
    ]
