#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_webrtc_sidecar_service.py"


def contract_fixture() -> dict:
    audio = {
        "sample_rate": 48000,
        "channels": 1,
        "frame_ms": 20,
        "encoding": "pcm_s16le",
        "bytes_per_sample": 2,
        "samples_per_frame": 960,
        "frame_bytes": 1920,
        "default_drain_bytes": 96000,
        "max_outbound_queue_bytes": 960000,
        "max_inbound_queue_bytes": 960000,
        "max_drain_wait_ms": 5000,
    }
    return {
        "contract": "voice.webrtc_sidecar",
        "version": 1,
        "audio": audio,
        "voice_surfaces": {
            "completed_voice_note": {
                "output": "audio/ogg; codecs=opus",
                "transport": "completed_file",
            },
            "streamed_voice_note": {
                "output": "audio/ogg; codecs=opus",
                "transport": "daemon_stream_encoded_file",
            },
            "raw_outbound_pcm": {
                "output": "pcm_s16le",
                "transport": "stdout_pcm_frames",
                "frame_bytes": 1920,
            },
            "raw_inbound_pcm": {
                "input": "pcm_s16le",
                "transport": "stdin_pcm_frames",
                "frame_bytes": 1920,
            },
            "file_transcription_smoke": {
                "input": "audio_file",
                "transport": "decoded_file_to_daemon_frames",
            },
        },
        "endpoints": {
            "contract": {"method": "GET", "path": "/contract"},
            "health": {"method": "GET", "path": "/health"},
            "offer": {"method": "POST", "path": "/offer"},
            "call_status": {"method": "GET", "path": "/calls/{call_id}"},
            "receive_audio": {"method": "GET", "path": "/calls/{call_id}/audio"},
            "send_audio": {"method": "POST", "path": "/calls/{call_id}/audio"},
            "clear_audio": {
                "method": "POST",
                "path": "/calls/{call_id}/audio/clear",
            },
            "close_call": {"method": "POST", "path": "/calls/{call_id}/close"},
        },
    }


class JsonHandler(BaseHTTPRequestHandler):
    contract: dict = {}

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/contract":
            self.respond(self.contract)
        elif self.path == "/health":
            self.respond(
                {
                    "ok": True,
                    "sessions": 0,
                    "call_ids": [],
                    "audio": self.contract["audio"],
                }
            )
        else:
            self.send_error(404)

    def log_message(self, *_args):
        return

    def respond(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SidecarServer:
    def __init__(self, contract: dict):
        self.contract = contract
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self):
        class Handler(JsonHandler):
            pass

        Handler.contract = self.contract
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_exc):
        assert self.server is not None
        assert self.thread is not None
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


class WebrtcSidecarVerifierTests(unittest.TestCase):
    def make_fake_voice(self, directory: Path, contract: dict) -> Path:
        contract_path = directory / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        voice_bin = directory / "voice"
        voice_bin.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"${1:-}\" == \"stream-contract\" ]]; then\n"
            f"  cat {str(contract_path)!r}\n"
            "else\n"
            "  echo \"unexpected command\" >&2\n"
            "  exit 64\n"
            "fi\n",
            encoding="utf-8",
        )
        voice_bin.chmod(0o755)
        return voice_bin

    def test_verifier_passes_against_matching_http_contract(self):
        contract = contract_fixture()
        with tempfile.TemporaryDirectory() as tmp, SidecarServer(contract) as url:
            voice_bin = self.make_fake_voice(Path(tmp), contract)

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--skip-systemd",
                    "--voice-bin",
                    str(voice_bin),
                    "--sidecar-url",
                    url,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("ok: voice WebRTC sidecar service verifier passed", result.stdout)
        self.assertIn("contract=matched", result.stdout)
        self.assertIn("systemd=skipped", result.stdout)

    def test_verifier_fails_when_sidecar_contract_differs_from_voice(self):
        voice_contract = contract_fixture()
        sidecar_contract = contract_fixture()
        sidecar_contract["audio"] = {**sidecar_contract["audio"], "frame_ms": 10}

        with tempfile.TemporaryDirectory() as tmp, SidecarServer(sidecar_contract) as url:
            voice_bin = self.make_fake_voice(Path(tmp), voice_contract)

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--skip-systemd",
                    "--voice-bin",
                    str(voice_bin),
                    "--sidecar-url",
                    url,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sidecar /contract does not match", result.stderr)
        self.assertIn("audio.frame_ms=10, expected 20", result.stderr)


if __name__ == "__main__":
    unittest.main()
