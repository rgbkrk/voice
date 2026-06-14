#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_whatsapp_inbound_audio_cache.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_whatsapp_inbound_audio_cache", SCRIPT_PATH
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
            print('{"event":"stt.transcribed","data":{"text":"hello from whatsapp","frames":12,"audio_duration_ms":240,"tokens":3}}')
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_audio(path: Path) -> None:
    path.write_bytes(b"OggS" + (b"\0" * 128))


def make_args(tmp_path: Path, **overrides):
    hermes_home = tmp_path / "hermes"
    audio_cache = hermes_home / "audio_cache"
    audio_cache.mkdir(parents=True)
    voice = tmp_path / "voice"
    write_fake_voice(voice)
    values = {
        "voice_bin": str(voice),
        "hermes_home": hermes_home,
        "audio_cache_dir": None,
        "audio_file": None,
        "max_files": 1,
        "require_cache": False,
        "run_stt": False,
        "skip_ffprobe": True,
        "timeout": 1.0,
        "stt_timeout": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class WhatsAppInboundAudioCacheVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script_module()

    def test_discovers_bridge_downloaded_audio_without_draining_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path)
            write_audio(args.hermes_home / "audio_cache" / "aud_abc123.ogg")
            write_audio(args.hermes_home / "audio_cache" / "tts_ignored.ogg")

            result = self.script.verify(args)

        self.assertTrue(result["success"], result["failures"])
        self.assertEqual(result["checks"]["discovered_count"], 1)
        self.assertTrue(result["checks"]["selected_files"][0].endswith("aud_abc123.ogg"))
        self.assertTrue(result["checks"]["audio"][0]["stt"]["skipped"])

    def test_run_stt_transcribes_cached_inbound_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, run_stt=True)
            write_audio(args.hermes_home / "audio_cache" / "aud_abc123.ogg")

            result = self.script.verify(args)

        self.assertTrue(result["success"], result["failures"])
        terminal = result["checks"]["audio"][0]["stt"]["terminal_event"]
        self.assertEqual(terminal["event"], "stt.transcribed")
        self.assertEqual(terminal["data"]["text"], "hello from whatsapp")

    def test_explicit_audio_file_must_look_like_bridge_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "not_bridge.ogg"
            write_audio(audio)
            args = make_args(tmp_path, audio_file=[audio])

            result = self.script.verify(args)

        self.assertFalse(result["success"])
        self.assertIn("must start with aud_", "\n".join(result["failures"]))

    def test_require_cache_fails_when_no_inbound_audio_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.script.verify(make_args(Path(tmp), require_cache=True))

        self.assertFalse(result["success"])
        self.assertIn("no bridge-downloaded inbound audio", "\n".join(result["failures"]))


if __name__ == "__main__":
    unittest.main()
