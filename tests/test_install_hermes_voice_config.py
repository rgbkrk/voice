#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "install_hermes_voice_config.py"
VERIFY_PATH = REPO_ROOT / "scripts" / "verify_hermes_voice_config.py"


def has_yaml() -> bool:
    try:
        import yaml  # noqa: F401
    except Exception:
        return False
    return True


def base_config() -> str:
    return """\
model:
  default: gpt-5.5
tts:
  provider: edge
  edge:
    voice: en-US-AriaNeural
  providers:
    legacy:
      type: command
      command: old-tts {input_path} {output_path}
      output_format: wav
stt:
  enabled: false
  provider: local
  providers:
    local:
      type: command
      command: old-stt {input_path}
      format: txt
"""


class HermesVoiceConfigInstallerTests(unittest.TestCase):
    def test_print_snippet_does_not_require_config_file(self):
        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--config",
                "/tmp/does-not-exist-hermes-config.yaml",
                "--voice-bin",
                "/opt/voice/bin/voice",
                "--print-snippet",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("tts:", result.stdout)
        self.assertIn("provider: kokoro", result.stdout)
        self.assertIn("/opt/voice/bin/voice say --format ogg-opus", result.stdout)
        self.assertIn("voice_compatible: true", result.stdout)
        self.assertIn("stt:", result.stdout)
        self.assertIn("/opt/voice/bin/voice stream-transcribe --quiet", result.stdout)

    @unittest.skipUnless(has_yaml(), "PyYAML is required for write-mode config tests")
    def test_dry_run_does_not_modify_config_and_verifies_temp_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(base_config(), encoding="utf-8")
            before = config.read_text(encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--voice-bin",
                    "/opt/voice/bin/voice",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            after = config.read_text(encoding="utf-8")

        self.assertEqual(after, before)
        self.assertIn("dry_run=true", result.stdout)
        self.assertIn("applied=false", result.stdout)
        self.assertIn("tts.provider=kokoro", result.stdout)
        self.assertIn("stt.provider=voice", result.stdout)
        self.assertIn("verify=passed", result.stdout)

    @unittest.skipUnless(has_yaml(), "PyYAML is required for write-mode config tests")
    def test_apply_patches_only_voice_provider_blocks_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(base_config(), encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--voice-bin",
                    "/opt/voice/bin/voice",
                    "--apply",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            verify = subprocess.run(
                [
                    str(VERIFY_PATH),
                    "--config",
                    str(config),
                    "--voice-bin",
                    "/opt/voice/bin/voice",
                    "--skip-tts-smoke",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            backups = list(Path(tmp).glob("config.yaml.bak.*"))
            rendered = config.read_text(encoding="utf-8")
            backup_text = backups[0].read_text(encoding="utf-8") if backups else ""

        self.assertIn("ok: Hermes voice config installed", result.stdout)
        self.assertIn("applied=true", result.stdout)
        self.assertIn("verify=passed", result.stdout)
        self.assertEqual(len(backups), 1)
        self.assertIn("provider: edge", backup_text)
        self.assertIn("default: gpt-5.5", rendered)
        self.assertIn("provider: kokoro", rendered)
        self.assertIn("/opt/voice/bin/voice say --format ogg-opus", rendered)
        self.assertIn("voice_compatible: true", rendered)
        self.assertIn("provider: voice", rendered)
        self.assertIn("/opt/voice/bin/voice stream-transcribe --quiet", rendered)
        self.assertIn("ok: Hermes voice config verifier passed", verify.stdout)

    @unittest.skipUnless(has_yaml(), "PyYAML is required for write-mode config tests")
    def test_apply_can_create_missing_config_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--voice-bin",
                    "voice",
                    "--apply",
                    "--create",
                    "--no-backup",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": os.environ.get("PATH", "")},
            )

            rendered = config.read_text(encoding="utf-8")

        self.assertIn("backup=none", result.stdout)
        self.assertIn("provider: kokoro", rendered)
        self.assertIn("provider: voice", rendered)

    @unittest.skipUnless(has_yaml(), "PyYAML is required for write-mode config tests")
    def test_dry_run_can_create_missing_config_shape_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--config",
                    str(config),
                    "--voice-bin",
                    "voice",
                    "--create",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertFalse(config.exists())
        self.assertIn("dry_run=true", result.stdout)
        self.assertIn("applied=false", result.stdout)
        self.assertIn("verify=passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
