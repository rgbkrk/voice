#!/usr/bin/env python3

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_whatsapp_voice_note_bridge.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_whatsapp_voice_note_bridge", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_fake_voice(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import pathlib
            import sys

            args = sys.argv[1:]
            output = pathlib.Path(args[args.index("--output") + 1])
            output.write_bytes(b"OggS" + (b"\\0" * 128))
            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def make_args(tmp_path: Path, **overrides):
    hermes_home = tmp_path / "hermes"
    env_file = hermes_home / ".env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("WHATSAPP_HOME_CHANNEL=20530681934008@lid\n", encoding="utf-8")
    voice = tmp_path / "voice"
    write_fake_voice(voice)
    values = {
        "voice_bin": str(voice),
        "hermes_home": hermes_home,
        "env_file": env_file,
        "bridge_url": "http://127.0.0.1:3000",
        "chat_id": None,
        "text": "Smoke.",
        "voice": "af_heart",
        "speed": "1.0",
        "output": None,
        "bridge_media_payload_js": None,
        "timeout": 1.0,
        "voice_timeout": 1.0,
        "skip_bridge_health": True,
        "skip_ffprobe": True,
        "skip_ptt_payload_check": True,
        "send": False,
        "wait_inbound_seconds": 0.0,
        "require_inbound_audio": False,
        "drain_bridge_messages": False,
        "keep_output": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class WhatsAppVoiceNoteBridgeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script_module()

    def test_dry_run_generates_ogg_without_sending(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.script.verify(make_args(Path(tmp)))

        self.assertTrue(result["success"], result["failures"])
        checks = result["checks"]
        self.assertEqual(checks["chat_id"], "20530681934008@lid")
        self.assertEqual(checks["voice_note"]["magic"], "OggS")
        self.assertTrue(checks["send_media"]["skipped"])

    def test_send_posts_generated_ogg_to_bridge_send_media(self):
        sent = []

        def fake_post(url, payload, *, timeout):
            sent.append((url, payload, timeout))
            return {"success": True, "messageId": "ABC123"}

        with tempfile.TemporaryDirectory() as tmp:
            result = self.script.verify(
                make_args(Path(tmp), send=True),
                post_json_func=fake_post,
            )

        self.assertTrue(result["success"], result["failures"])
        self.assertEqual(len(sent), 1)
        url, payload, timeout = sent[0]
        self.assertEqual(url, "http://127.0.0.1:3000/send-media")
        self.assertEqual(payload["chatId"], "20530681934008@lid")
        self.assertEqual(payload["mediaType"], "audio")
        self.assertTrue(os.path.isabs(payload["filePath"]))
        self.assertEqual(timeout, 1.0)
        self.assertEqual(
            result["checks"]["send_media"]["response"],
            {"success": True, "messageId": "ABC123"},
        )

    def test_send_requires_chat_id_or_home_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp), send=True)
            args.env_file.write_text("", encoding="utf-8")

            result = self.script.verify(args, post_json_func=lambda *a, **k: {})

        self.assertFalse(result["success"])
        self.assertIn("no chat id configured", "\n".join(result["failures"]))

    def test_wait_inbound_requires_explicit_message_drain(self):
        get_calls = []

        def fake_get(url, *, timeout):
            get_calls.append((url, timeout))
            return []

        with tempfile.TemporaryDirectory() as tmp:
            result = self.script.verify(
                make_args(
                    Path(tmp),
                    wait_inbound_seconds=1.0,
                    require_inbound_audio=True,
                ),
                get_json_func=fake_get,
            )

        self.assertFalse(result["success"])
        self.assertEqual(get_calls, [])
        self.assertIn("drains the bridge /messages queue", "\n".join(result["failures"]))
        self.assertEqual(
            result["checks"]["inbound_audio"]["reason"],
            "requires --drain-bridge-messages",
        )

    def test_wait_inbound_audio_polls_messages_when_drain_allowed(self):
        get_calls = []

        def fake_get(url, *, timeout):
            get_calls.append((url, timeout))
            return [
                {
                    "chatId": "20530681934008@lid",
                    "senderId": "13236478455@s.whatsapp.net",
                    "hasMedia": True,
                    "mediaType": "ptt",
                    "mediaUrls": ["/home/ubuntu/.hermes/audio_cache/aud_test.ogg"],
                }
            ]

        with tempfile.TemporaryDirectory() as tmp:
            result = self.script.verify(
                make_args(
                    Path(tmp),
                    wait_inbound_seconds=1.0,
                    require_inbound_audio=True,
                    drain_bridge_messages=True,
                ),
                get_json_func=fake_get,
            )

        self.assertTrue(result["success"], result["failures"])
        self.assertEqual(get_calls, [("http://127.0.0.1:3000/messages", 1.0)])
        inbound = result["checks"]["inbound_audio"]
        self.assertTrue(inbound["drains_bridge_messages"])
        self.assertEqual(len(inbound["seen_events"]), 1)
        self.assertEqual(len(inbound["audio_events"]), 1)


if __name__ == "__main__":
    unittest.main()
