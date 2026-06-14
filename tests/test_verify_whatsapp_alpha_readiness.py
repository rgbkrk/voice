#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_whatsapp_alpha_readiness.py"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def json_script(path: Path, payload: dict) -> None:
    payload_json = json.dumps(payload)
    write_executable(
        path,
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            print(json.dumps(json.loads({payload_json!r})))
            """
        ),
    )


def ok_shell(path: Path) -> None:
    write_executable(
        path,
        "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n",
    )


def write_fake_helpers(directory: Path, *, cloud_configured: bool = False) -> None:
    ok_shell(directory / "verify_hermes_voice_config.py")
    ok_shell(directory / "verify_whatsapp_voice_contract.sh")
    success_payload = {"success": True, "checks": {}, "failures": []}
    json_script(directory / "verify_hermes_gateway_service.py", success_payload)
    json_script(directory / "verify_cli_mcp_surface.py", success_payload)
    json_script(directory / "verify_webrtc_sidecar_service.py", success_payload)
    json_script(directory / "verify_whatsapp_voice_note_bridge.py", success_payload)
    json_script(directory / "verify_whatsapp_inbound_audio_cache.py", success_payload)

    cloud_missing = [] if cloud_configured else [
        "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
        "WHATSAPP_CLOUD_ACCESS_TOKEN",
        "WHATSAPP_CLOUD_APP_SECRET",
        "WHATSAPP_CLOUD_VERIFY_TOKEN",
    ]
    bridge_payload = {
        "success": True,
        "checks": {
            "whatsapp_cloud": {
                "cloud_configured": cloud_configured,
                "calling_sidecar_configured": True,
                "calling_ready": cloud_configured,
                "cloud_missing": cloud_missing,
                "calling_missing": cloud_missing,
            }
        },
        "failures": [],
    }
    json_script(directory / "verify_whatsapp_bridge_runtime.py", bridge_payload)


class WhatsAppAlphaReadinessTests(unittest.TestCase):
    def run_invalid(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT_PATH), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def run_readiness(
        self,
        tmp_path: Path,
        *args: str,
        skip_voice_note_smoke: bool = True,
    ) -> dict:
        helpers = tmp_path / "helpers"
        helpers.mkdir()
        write_fake_helpers(helpers)
        voice = tmp_path / "voice"
        write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
        config = tmp_path / "config.yaml"
        config.write_text("tts: {}\n", encoding="utf-8")
        command = [
            str(SCRIPT_PATH),
            "--voice-bin",
            str(voice),
            "--hermes-home",
            str(tmp_path / "hermes"),
            "--hermes-config",
            str(config),
            "--skip-systemd",
            "--skip-daemon",
            "--skip-sidecar",
            "--json",
        ]
        if skip_voice_note_smoke:
            command.append("--skip-voice-note-smoke")
        command.extend(args)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "VOICE_READINESS_SCRIPT_DIR": str(helpers),
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_readiness_succeeds_for_baileys_alpha_when_cloud_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(Path(tmp))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["external_meta_setup"]["cloud_configured"], False)
        self.assertIn(
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            payload["external_meta_setup"]["cloud_missing"],
        )
        self.assertTrue(payload["external_meta_setup"]["setup_steps"])

    def test_inbound_cache_smoke_adds_receive_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(
                Path(tmp),
                "--run-inbound-cache-smoke",
                "--whatsapp-audio-cache-dir",
                str(Path(tmp) / "audio_cache"),
            )

        self.assertTrue(payload["success"])
        components = {item["name"]: item for item in payload["components"]}
        self.assertIn("whatsapp_inbound_cache_stt", components)
        inbound = components["whatsapp_inbound_cache_stt"]
        self.assertEqual(inbound["category"], "voice_note")
        self.assertIn("--require-cache", inbound["command"])
        self.assertIn("--run-stt", inbound["command"])
        self.assertIn(
            "whatsapp_inbound_cache_stt",
            payload["by_category"]["voice_note"]["components"],
        )

    def test_voice_note_send_receive_flags_are_passed_to_bridge_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(
                Path(tmp),
                "--send-voice-note",
                "--voice-note-chat-id",
                "20530681934008@lid",
                "--wait-inbound-seconds",
                "5",
                "--require-inbound-audio",
                "--drain-bridge-messages",
                skip_voice_note_smoke=False,
            )

        self.assertTrue(payload["success"])
        components = {item["name"]: item for item in payload["components"]}
        self.assertIn("whatsapp_voice_note_send_receive", components)
        command = components["whatsapp_voice_note_send_receive"]["command"]
        self.assertIn("--send", command)
        self.assertIn("--chat-id", command)
        self.assertIn("20530681934008@lid", command)
        self.assertIn("--wait-inbound-seconds", command)
        self.assertIn("5.0", command)
        self.assertIn("--require-inbound-audio", command)
        self.assertIn("--drain-bridge-messages", command)

    def test_voice_note_flags_cannot_be_used_when_voice_note_smoke_is_skipped(self):
        result = self.run_invalid("--skip-voice-note-smoke", "--send-voice-note")

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be used with --skip-voice-note-smoke", result.stderr)

    def test_wait_inbound_requires_explicit_drain_flag(self):
        result = self.run_invalid("--wait-inbound-seconds", "5")

        self.assertEqual(result.returncode, 2)
        self.assertIn("add --drain-bridge-messages", result.stderr)

    def test_require_inbound_audio_requires_wait_window(self):
        result = self.run_invalid("--require-inbound-audio")

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--require-inbound-audio requires --wait-inbound-seconds",
            result.stderr,
        )

    def test_voice_note_chat_id_requires_real_send(self):
        result = self.run_invalid("--voice-note-chat-id", "20530681934008@lid")

        self.assertEqual(result.returncode, 2)
        self.assertIn("--voice-note-chat-id requires --send-voice-note", result.stderr)

    def test_require_cloud_calling_fails_when_meta_credentials_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            helpers = tmp_path / "helpers"
            helpers.mkdir()
            write_fake_helpers(helpers, cloud_configured=False)
            voice = tmp_path / "voice"
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config = tmp_path / "config.yaml"
            config.write_text("tts: {}\n", encoding="utf-8")
            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-home",
                    str(tmp_path / "hermes"),
                    "--hermes-config",
                    str(config),
                    "--skip-systemd",
                    "--skip-daemon",
                    "--skip-sidecar",
                    "--skip-voice-note-smoke",
                    "--require-whatsapp-calling",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "VOICE_READINESS_SCRIPT_DIR": str(helpers),
                },
            )

        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["success"])
        self.assertIn(
            "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
            payload["external_meta_setup"]["calling_missing"],
        )


if __name__ == "__main__":
    unittest.main()
