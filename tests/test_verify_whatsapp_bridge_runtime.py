#!/usr/bin/env python3

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_whatsapp_bridge_runtime.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_whatsapp_bridge_runtime", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_baileys_session(session_dir: Path) -> None:
    session_dir.mkdir(parents=True)
    write_json(
        session_dir / "creds.json",
        {
            "me": {
                "id": "13236478455:2@s.whatsapp.net",
                "name": "Quill",
                "lid": "186999436771390:2@lid",
            },
            "platform": "smbi",
            "accountSyncCounter": 1,
        },
    )
    write_json(session_dir / "pre-key-1.json", {})
    write_json(session_dir / "session-186999436771390_1.0.json", {})
    write_json(session_dir / "app-state-sync-key-AAAA.json", {})
    write_json(session_dir / "lid-mapping-13236478455.json", "186999436771390")
    write_json(session_dir / "lid-mapping-186999436771390_reverse.json", "13236478455")


def make_args(tmp_path: Path, **overrides):
    hermes_home = tmp_path / "hermes"
    session_dir = hermes_home / "whatsapp" / "session"
    env_file = hermes_home / ".env"
    values = {
        "hermes_home": hermes_home,
        "session_dir": session_dir,
        "env_file": env_file,
        "bridge_url": "http://127.0.0.1:3000",
        "service_name": "hermes-gateway.service",
        "expected_agent_number": "13236478455",
        "expected_agent_name": "Quill",
        "expected_mode": None,
        "timeout": 1.0,
        "skip_bridge_health": True,
        "skip_process_check": True,
        "skip_systemd": True,
        "require_whatsapp_cloud": False,
        "require_whatsapp_calling": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class WhatsAppBridgeRuntimeVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script_module()

    def test_baileys_identity_passes_and_reports_missing_cloud_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text(
                "WHATSAPP_ENABLED=true\nWHATSAPP_MODE=bot\n",
                encoding="utf-8",
            )

            result = self.script.verify(args)

        self.assertTrue(result["success"], result["failures"])
        checks = result["checks"]
        self.assertEqual(checks["baileys_identity"]["name"], "Quill")
        self.assertEqual(checks["baileys_identity"]["number"], "13236478455")
        self.assertEqual(checks["baileys_identity"]["lid_number"], "186999436771390")
        self.assertTrue(checks["lid_mapping"]["ok"])
        self.assertFalse(checks["whatsapp_cloud"]["cloud_configured"])
        self.assertIn(
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            checks["whatsapp_cloud"]["cloud_missing"],
        )

    def test_expected_agent_number_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, expected_agent_number="15551234567")
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text("WHATSAPP_MODE=bot\n", encoding="utf-8")

            result = self.script.verify(args)

        self.assertFalse(result["success"])
        self.assertIn("does not match expected", "\n".join(result["failures"]))

    def test_require_cloud_fails_when_cloud_credentials_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, require_whatsapp_cloud=True)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text("WHATSAPP_MODE=bot\n", encoding="utf-8")

            result = self.script.verify(args)

        self.assertFalse(result["success"])
        self.assertIn("WhatsApp Cloud credentials missing", "\n".join(result["failures"]))

    def test_cloud_calling_summary_merges_env_file_and_systemd_sources(self):
        summary = self.script.build_cloud_summary(
            {
                "env_file": {
                    "WHATSAPP_CLOUD_PHONE_NUMBER_ID": "phone-id",
                    "WHATSAPP_CLOUD_ACCESS_TOKEN": "token",
                    "WHATSAPP_CLOUD_APP_SECRET": "secret",
                    "WHATSAPP_CLOUD_VERIFY_TOKEN": "verify",
                },
                "systemd_service": {
                    "WHATSAPP_CLOUD_CALLING_SIDECAR_URL": "http://127.0.0.1:8787",
                    "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND": "voice stream",
                },
            }
        )

        self.assertTrue(summary["cloud_configured"])
        self.assertTrue(summary["calling_sidecar_configured"])
        self.assertTrue(summary["calling_ready"])
        self.assertEqual(summary["calling_missing"], [])

    def test_cloud_calling_summary_can_use_running_gateway_process_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, skip_systemd=False)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text("WHATSAPP_MODE=bot\n", encoding="utf-8")

            original_service_env = self.script.get_systemd_service_env
            original_main_pid = self.script.get_systemd_main_pid
            original_process_env = self.script.read_process_environment
            self.script.get_systemd_service_env = lambda *_args, **_kwargs: ({}, None)
            self.script.get_systemd_main_pid = lambda *_args, **_kwargs: (4242, None)
            self.script.read_process_environment = lambda _pid: (
                {
                    "WHATSAPP_CLOUD_PHONE_NUMBER_ID": "phone-id",
                    "WHATSAPP_CLOUD_ACCESS_TOKEN": "token",
                    "WHATSAPP_CLOUD_APP_SECRET": "secret",
                    "WHATSAPP_CLOUD_VERIFY_TOKEN": "verify",
                    "WHATSAPP_CLOUD_CALLING_SIDECAR_URL": "http://127.0.0.1:8787",
                    "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND": "voice stream",
                },
                None,
            )
            try:
                result = self.script.verify(args)
            finally:
                self.script.get_systemd_service_env = original_service_env
                self.script.get_systemd_main_pid = original_main_pid
                self.script.read_process_environment = original_process_env

        self.assertTrue(result["success"], result["failures"])
        checks = result["checks"]
        self.assertTrue(checks["whatsapp_cloud"]["cloud_configured"])
        self.assertTrue(checks["whatsapp_cloud"]["calling_ready"])
        self.assertIn(
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            checks["env_key_sources"]["systemd_process"],
        )
        self.assertNotIn("token", json.dumps(checks["env_key_sources"]))


if __name__ == "__main__":
    unittest.main()
