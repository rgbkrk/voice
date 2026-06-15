#!/usr/bin/env python3

import argparse
import hashlib
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
        "check_whatsapp_cloud_api": False,
        "graph_api_base_url": "https://graph.facebook.com",
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
                "\n".join(
                    [
                        "WHATSAPP_ENABLED=true",
                        "WHATSAPP_MODE=bot",
                        "WHATSAPP_HOME_CHANNEL=20530681934008@lid",
                        "WHATSAPP_ALLOWED_USERS=18316653748,17202993514",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            result = self.script.verify(args)

        self.assertTrue(result["success"], result["failures"])
        checks = result["checks"]
        self.assertEqual(checks["baileys_identity"]["name"], "Quill")
        self.assertEqual(checks["baileys_identity"]["number"], "13236478455")
        self.assertEqual(checks["baileys_identity"]["lid_number"], "186999436771390")
        self.assertTrue(checks["lid_mapping"]["ok"])
        local = checks["whatsapp_local_config"]
        self.assertTrue(local["enabled"])
        self.assertEqual(local["mode"], "bot")
        self.assertEqual(local["home_channel"], "20530681934008@lid")
        self.assertEqual(local["home_channel_kind"], "lid")
        self.assertEqual(local["allowed_users_count"], 2)
        self.assertNotIn("18316653748", json.dumps(local))
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

    def test_bridge_script_hash_matches_health_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(
                tmp_path,
                skip_bridge_health=False,
                skip_process_check=False,
            )
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text("WHATSAPP_MODE=bot\n", encoding="utf-8")
            bridge_script = tmp_path / "whatsapp-bridge" / "bridge.js"
            bridge_script.parent.mkdir()
            bridge_script.write_text("console.log('bridge');\n", encoding="utf-8")
            script_hash = hashlib.sha256(bridge_script.read_bytes()).hexdigest()[:16]

            original_fetch = self.script.fetch_bridge_health
            original_processes = self.script.find_bridge_processes
            self.script.fetch_bridge_health = lambda *_args, **_kwargs: (
                {
                    "status": "connected",
                    "queueLength": 0,
                    "scriptHash": script_hash,
                },
                None,
            )
            self.script.find_bridge_processes = lambda **_kwargs: [
                {
                    "pid": 1234,
                    "script": str(bridge_script),
                    "port": "3000",
                    "session": str(args.session_dir),
                    "mode": "bot",
                }
            ]
            try:
                result = self.script.verify(args)
            finally:
                self.script.fetch_bridge_health = original_fetch
                self.script.find_bridge_processes = original_processes

        self.assertTrue(result["success"], result["failures"])
        hash_check = result["checks"]["bridge_script_hash"]
        self.assertTrue(hash_check["checked"])
        self.assertTrue(hash_check["ok"])
        self.assertEqual(hash_check["reported"], script_hash)
        self.assertEqual(hash_check["computed"], script_hash)

    def test_bridge_script_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(
                tmp_path,
                skip_bridge_health=False,
                skip_process_check=False,
            )
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text("WHATSAPP_MODE=bot\n", encoding="utf-8")
            bridge_script = tmp_path / "whatsapp-bridge" / "bridge.js"
            bridge_script.parent.mkdir()
            bridge_script.write_text("console.log('new bridge');\n", encoding="utf-8")

            original_fetch = self.script.fetch_bridge_health
            original_processes = self.script.find_bridge_processes
            self.script.fetch_bridge_health = lambda *_args, **_kwargs: (
                {
                    "status": "connected",
                    "queueLength": 0,
                    "scriptHash": "stale00000000000",
                },
                None,
            )
            self.script.find_bridge_processes = lambda **_kwargs: [
                {
                    "pid": 1234,
                    "script": str(bridge_script),
                    "port": "3000",
                    "session": str(args.session_dir),
                    "mode": "bot",
                }
            ]
            try:
                result = self.script.verify(args)
            finally:
                self.script.fetch_bridge_health = original_fetch
                self.script.find_bridge_processes = original_processes

        self.assertFalse(result["success"])
        self.assertIn("script hash mismatch", "\n".join(result["failures"]))
        hash_check = result["checks"]["bridge_script_hash"]
        self.assertTrue(hash_check["checked"])
        self.assertFalse(hash_check["ok"])
        self.assertEqual(hash_check["reported"], "stale00000000000")

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

    def test_disabled_local_whatsapp_config_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text(
                "WHATSAPP_ENABLED=false\nWHATSAPP_MODE=bot\n",
                encoding="utf-8",
            )

            result = self.script.verify(args)

        self.assertFalse(result["success"])
        self.assertIn("WHATSAPP_ENABLED is explicitly disabled", result["failures"])

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
        self.assertEqual(summary["cloud_invalid"], [])
        self.assertEqual(summary["webhook"]["host"], "0.0.0.0")
        self.assertEqual(summary["webhook"]["port"], 8090)
        self.assertEqual(summary["webhook"]["path"], "/whatsapp/webhook")
        self.assertEqual(summary["webhook"]["api_version"], "v20.0")
        self.assertIn(
            "WHATSAPP_CLOUD_WEBHOOK_PORT",
            summary["webhook"]["defaulted"],
        )

    def test_invalid_cloud_webhook_config_fails_strict_cloud_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, require_whatsapp_cloud=True)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text(
                "\n".join(
                    [
                        "WHATSAPP_MODE=bot",
                        "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                        "WHATSAPP_CLOUD_ACCESS_TOKEN=secret-token",
                        "WHATSAPP_CLOUD_APP_SECRET=secret-app",
                        "WHATSAPP_CLOUD_VERIFY_TOKEN=secret-verify",
                        "WHATSAPP_CLOUD_WEBHOOK_PORT=not-a-port",
                        "WHATSAPP_CLOUD_WEBHOOK_PATH=whatsapp/webhook",
                        "WHATSAPP_CLOUD_API_VERSION=20",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            result = self.script.verify(args)

        self.assertFalse(result["success"])
        failures = "\n".join(result["failures"])
        self.assertIn("WhatsApp Cloud config invalid", failures)
        self.assertIn("WHATSAPP_CLOUD_WEBHOOK_PORT", failures)
        self.assertIn("WHATSAPP_CLOUD_WEBHOOK_PATH", failures)
        self.assertIn("WHATSAPP_CLOUD_API_VERSION", failures)
        self.assertNotIn("secret-token", json.dumps(result))
        webhook = result["checks"]["whatsapp_cloud"]["webhook"]
        self.assertEqual(webhook["port"], None)
        self.assertEqual(
            webhook["invalid"],
            [
                "WHATSAPP_CLOUD_WEBHOOK_PORT",
                "WHATSAPP_CLOUD_WEBHOOK_PATH",
                "WHATSAPP_CLOUD_API_VERSION",
            ],
        )

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

    def test_cloud_api_phone_number_check_uses_safe_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, check_whatsapp_cloud_api=True)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text(
                "\n".join(
                    [
                        "WHATSAPP_MODE=bot",
                        "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                        "WHATSAPP_CLOUD_ACCESS_TOKEN=secret-token",
                        "WHATSAPP_CLOUD_APP_SECRET=secret-app",
                        "WHATSAPP_CLOUD_VERIFY_TOKEN=secret-verify",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            original_fetch = self.script.fetch_cloud_phone_number
            calls = []

            def fake_fetch(**kwargs):
                calls.append(kwargs)
                return (
                    {
                        "id": "7794189252778687",
                        "display_phone_number": "+1 555 0100",
                        "verified_name": "Quill",
                        "quality_rating": "GREEN",
                        "platform_type": "CLOUD_API",
                        "throughput": {"level": "STANDARD"},
                    },
                    200,
                    None,
                )

            self.script.fetch_cloud_phone_number = fake_fetch
            try:
                result = self.script.verify(args)
            finally:
                self.script.fetch_cloud_phone_number = original_fetch

        self.assertTrue(result["success"], result["failures"])
        self.assertEqual(calls[0]["phone_number_id"], "7794189252778687")
        self.assertEqual(calls[0]["access_token"], "secret-token")
        cloud_api = result["checks"]["whatsapp_cloud"]["cloud_api"]
        self.assertTrue(cloud_api["checked"])
        self.assertTrue(cloud_api["ok"])
        self.assertEqual(cloud_api["http_status"], 200)
        phone = cloud_api["phone_number"]
        self.assertTrue(phone["id_matches_config"])
        self.assertTrue(phone["display_phone_number_present"])
        self.assertTrue(phone["verified_name_present"])
        self.assertEqual(phone["quality_rating"], "GREEN")
        self.assertEqual(phone["platform_type"], "CLOUD_API")
        self.assertEqual(phone["throughput_level"], "STANDARD")
        serialized = json.dumps(result)
        self.assertNotIn("secret-token", serialized)
        self.assertNotIn("+1 555 0100", serialized)
        self.assertNotIn('"Quill"', json.dumps(cloud_api))

    def test_cloud_api_check_fails_without_required_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, check_whatsapp_cloud_api=True)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text("WHATSAPP_MODE=bot\n", encoding="utf-8")

            result = self.script.verify(args)

        self.assertFalse(result["success"])
        cloud_api = result["checks"]["whatsapp_cloud"]["cloud_api"]
        self.assertTrue(cloud_api["checked"])
        self.assertFalse(cloud_api["ok"])
        self.assertEqual(
            cloud_api["missing"],
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
                "WHATSAPP_CLOUD_ACCESS_TOKEN",
            ],
        )
        self.assertIn(
            "WhatsApp Cloud API phone number check failed",
            "\n".join(result["failures"]),
        )

    def test_cloud_api_http_error_is_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path, check_whatsapp_cloud_api=True)
            write_baileys_session(args.session_dir)
            args.env_file.parent.mkdir(parents=True, exist_ok=True)
            args.env_file.write_text(
                "\n".join(
                    [
                        "WHATSAPP_MODE=bot",
                        "WHATSAPP_CLOUD_PHONE_NUMBER_ID=7794189252778687",
                        "WHATSAPP_CLOUD_ACCESS_TOKEN=secret-token",
                        "WHATSAPP_CLOUD_APP_SECRET=secret-app",
                        "WHATSAPP_CLOUD_VERIFY_TOKEN=secret-verify",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            original_fetch = self.script.fetch_cloud_phone_number

            def fake_fetch(**_kwargs):
                return (
                    None,
                    403,
                    {
                        "message": "Unsupported get request",
                        "type": "GraphMethodException",
                        "code": 100,
                    },
                )

            self.script.fetch_cloud_phone_number = fake_fetch
            try:
                result = self.script.verify(args)
            finally:
                self.script.fetch_cloud_phone_number = original_fetch

        self.assertFalse(result["success"])
        cloud_api = result["checks"]["whatsapp_cloud"]["cloud_api"]
        self.assertFalse(cloud_api["ok"])
        self.assertEqual(cloud_api["http_status"], 403)
        self.assertEqual(cloud_api["error"]["code"], 100)
        self.assertNotIn("secret-token", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
