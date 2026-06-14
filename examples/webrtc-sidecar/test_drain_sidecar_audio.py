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


def load_drain():
    load_bridge_module()
    path = Path(__file__).with_name("drain_sidecar_audio.py")
    spec = importlib.util.spec_from_file_location("voice_webrtc_drain_sidecar_audio", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audio_contract():
    bridge = sys.modules["post_voice_stream"]
    return bridge.AudioContract(
        sample_rate=48_000,
        channels=1,
        frame_ms=20,
        encoding="pcm_s16le",
        frame_bytes=1_920,
        default_drain_bytes=96_000,
        max_drain_wait_ms=5_000,
    )


def test_drain_url_escapes_call_id_and_uses_audio_query():
    drain = load_drain()

    url = drain.drain_url("http://127.0.0.1:8787/", "wamid/call 1", 1_920, 500)

    assert (
        url
        == "http://127.0.0.1:8787/calls/wamid%2Fcall%201/audio?max_bytes=1920&wait_ms=500"
    )


def test_validate_drain_shape_requires_whole_webrtc_frames():
    drain = load_drain()
    contract = audio_contract()

    drain.validate_drain_shape(contract, 1_920, 500)
    drain.validate_drain_shape(contract, 3_840, 0)

    try:
        drain.validate_drain_shape(contract, 960, 500)
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("expected non-frame-aligned drain size to be rejected")

    try:
        drain.validate_drain_shape(contract, 1_920, -1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected negative wait_ms to be rejected")


def test_drain_defaults_use_contract_window_and_cap_wait():
    drain = load_drain()
    contract = audio_contract()

    assert drain.default_max_bytes(contract, None) == 96_000
    assert drain.default_max_bytes(contract, 1_920) == 1_920
    assert drain.capped_wait_ms(contract, 250) == 250
    assert drain.capped_wait_ms(contract, 10_000) == 5_000


def test_decode_audio_response_validates_returned_bytes():
    drain = load_drain()

    pcm = drain.decode_audio_response(
        {
            "returned_bytes": 2,
            "pcm_s16le_base64": base64.b64encode(b"\x01\x00").decode("ascii"),
        }
    )

    assert pcm == b"\x01\x00"

    try:
        drain.decode_audio_response(
            {
                "returned_bytes": 4,
                "pcm_s16le_base64": base64.b64encode(b"\x01\x00").decode("ascii"),
            }
        )
    except ValueError as exc:
        assert "returned_bytes" in str(exc)
    else:
        raise AssertionError("expected returned byte mismatch to be rejected")


def test_fetch_audio_chunk_reads_sidecar_json_response():
    drain = load_drain()
    received_paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received_paths.append(self.path)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "returned_bytes": 2,
                        "pcm_s16le_base64": base64.b64encode(b"\x02\x00").decode(
                            "ascii"
                        ),
                    }
                ).encode("utf-8")
            )

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/calls/call-1/audio?max_bytes=1920&wait_ms=1"

        pcm = drain.fetch_audio_chunk(url, 1.0)

        assert pcm == b"\x02\x00"
        assert received_paths == ["/calls/call-1/audio?max_bytes=1920&wait_ms=1"]
    finally:
        server.shutdown()
        thread.join(timeout=1)
