from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import threading


def load_bridge_module():
    path = Path(__file__).with_name("post_voice_stream.py")
    spec = importlib.util.spec_from_file_location("post_voice_stream", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_drain_module():
    load_bridge_module()
    path = Path(__file__).with_name("drain_sidecar_audio.py")
    spec = importlib.util.spec_from_file_location("drain_sidecar_audio", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_echo():
    load_drain_module()
    path = Path(__file__).with_name("echo_sidecar_audio.py")
    spec = importlib.util.spec_from_file_location("voice_webrtc_echo_sidecar_audio", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_max_bytes_prefers_one_webrtc_frame():
    echo = load_echo()

    assert echo.default_max_bytes(None, 1_920) == 1_920
    assert echo.default_max_bytes(3_840, 1_920) == 3_840


def test_echo_chunks_drains_and_posts_same_pcm(monkeypatch):
    echo = load_echo()
    bridge = sys.modules["post_voice_stream"]
    contract = bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
        default_drain_bytes=96_000,
        max_outbound_queue_bytes=960_000,
        max_drain_wait_ms=5_000,
    )
    monkeypatch.setattr(echo, "load_audio_contract", lambda: contract)

    received_posts: list[tuple[str, dict[str, object]]] = []
    get_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal get_count
            get_count += 1
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            pcm = b"\x01\x00\xff\xff" if get_count == 1 else b""
            self.wfile.write(
                json.dumps(
                    {
                        "returned_bytes": len(pcm),
                        "pcm_s16le_base64": base64.b64encode(pcm).decode("ascii"),
                    }
                ).encode("utf-8")
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            received_posts.append((self.path, body))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"accepted_bytes":4}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        sidecar_url = f"http://127.0.0.1:{server.server_port}"
        exit_code = echo.echo_chunks(
            sidecar_url=sidecar_url,
            call_id="call-1",
            max_bytes=None,
            wait_ms=1,
            duration_ms=None,
            stop_after_empty=1,
            timeout_s=1.0,
            quiet=True,
        )

        assert exit_code == 0
        assert get_count == 2
        assert received_posts == [
            (
                "/calls/call-1/audio",
                {
                    "sample_rate": 48_000,
                    "channels": 1,
                    "frame_ms": 20,
                    "encoding": "pcm_s16le",
                    "pcm_s16le_base64": "AQD//w==",
                },
            )
        ]
    finally:
        server.shutdown()
        thread.join(timeout=1)


def test_echo_chunks_returns_tempfail_on_backpressure(monkeypatch):
    echo = load_echo()
    bridge = sys.modules["post_voice_stream"]
    contract = bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
        default_drain_bytes=96_000,
        max_outbound_queue_bytes=960_000,
        max_drain_wait_ms=5_000,
    )
    monkeypatch.setattr(echo, "load_audio_contract", lambda: contract)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            pcm = b"\x01\x00"
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "returned_bytes": len(pcm),
                        "pcm_s16le_base64": base64.b64encode(pcm).decode("ascii"),
                    }
                ).encode("utf-8")
            )

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
        exit_code = echo.echo_chunks(
            sidecar_url=f"http://127.0.0.1:{server.server_port}",
            call_id="call-1",
            max_bytes=None,
            wait_ms=1,
            duration_ms=None,
            stop_after_empty=1,
            timeout_s=1.0,
            quiet=True,
        )

        assert exit_code == echo.TEMPFAIL_EXIT_CODE
    finally:
        server.shutdown()
        thread.join(timeout=1)
