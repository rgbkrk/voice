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


def write_fake_helpers(
    directory: Path,
    *,
    cloud_configured: bool = False,
    voice_note_payload: dict | None = None,
    inbound_cache_payload: dict | None = None,
) -> None:
    ok_shell(directory / "verify_hermes_voice_config.py")
    ok_shell(directory / "verify_whatsapp_voice_contract.sh")
    success_payload = {"success": True, "checks": {}, "failures": []}
    json_script(directory / "verify_hermes_gateway_service.py", success_payload)
    json_script(directory / "verify_cli_mcp_surface.py", success_payload)
    json_script(directory / "verify_webrtc_sidecar_service.py", success_payload)
    json_script(
        directory / "verify_whatsapp_voice_note_bridge.py",
        voice_note_payload or success_payload,
    )
    json_script(
        directory / "verify_whatsapp_inbound_audio_cache.py",
        inbound_cache_payload or success_payload,
    )

    cloud_missing = [] if cloud_configured else [
        "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
        "WHATSAPP_CLOUD_ACCESS_TOKEN",
        "WHATSAPP_CLOUD_APP_SECRET",
        "WHATSAPP_CLOUD_VERIFY_TOKEN",
    ]
    cloud_required = {
        key: {
            "present": key not in cloud_missing,
            "sources": ["env_file"] if key not in cloud_missing else [],
        }
        for key in [
            "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            "WHATSAPP_CLOUD_APP_SECRET",
            "WHATSAPP_CLOUD_VERIFY_TOKEN",
        ]
    }
    calling = {
        "WHATSAPP_CLOUD_CALLING_SIDECAR_URL": {
            "present": True,
            "sources": ["systemd_service"],
        },
        "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND": {
            "present": True,
            "sources": ["systemd_service"],
        },
        "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT": {
            "present": True,
            "sources": ["systemd_service"],
        },
    }
    env_key_sources = {
        "env_file": [
            "WHATSAPP_ENABLED",
            "WHATSAPP_HOME_CHANNEL",
            "WHATSAPP_MODE",
            *([] if not cloud_configured else list(cloud_required)),
        ],
        "systemd_service": list(calling),
    }
    bridge_payload = {
        "success": True,
        "checks": {
            "env_file": str(directory.parent / "hermes" / ".env"),
            "env_key_sources": env_key_sources,
            "baileys_identity": {
                "name": "Quill",
                "number": "13236478455",
                "lid_number": "186999436771390",
            },
            "whatsapp_local_config": {
                "home_channel": "20530681934008@lid",
                "home_channel_kind": "lid",
                "mode": "bot",
                "allowed_users_count": 2,
            },
            "whatsapp_cloud": {
                "cloud_configured": cloud_configured,
                "calling_sidecar_configured": True,
                "calling_ready": cloud_configured,
                "cloud_missing": cloud_missing,
                "calling_missing": cloud_missing,
                "cloud_required": cloud_required,
                "calling": calling,
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
        cloud_configured: bool = False,
        skip_voice_note_smoke: bool = True,
        voice_note_payload: dict | None = None,
        inbound_cache_payload: dict | None = None,
    ) -> dict:
        result = self.run_readiness_process(
            tmp_path,
            *args,
            cloud_configured=cloud_configured,
            skip_voice_note_smoke=skip_voice_note_smoke,
            voice_note_payload=voice_note_payload,
            inbound_cache_payload=inbound_cache_payload,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def run_readiness_process(
        self,
        tmp_path: Path,
        *args: str,
        cloud_configured: bool = False,
        skip_voice_note_smoke: bool = True,
        voice_note_payload: dict | None = None,
        inbound_cache_payload: dict | None = None,
    ) -> subprocess.CompletedProcess[str]:
        helpers = tmp_path / "helpers"
        helpers.mkdir()
        write_fake_helpers(
            helpers,
            cloud_configured=cloud_configured,
            voice_note_payload=voice_note_payload,
            inbound_cache_payload=inbound_cache_payload,
        )
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
        return result

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
        gates = payload["pending_gates"]
        self.assertEqual(
            gates["attended_fresh_receive"]["status"],
            "pending_attended",
        )
        self.assertFalse(gates["attended_fresh_receive"]["drains_bridge_messages"])
        self.assertIn(
            "attended-cache-receive",
            gates["attended_fresh_receive"]["command"],
        )
        self.assertIn(
            "attended-send-receive",
            gates["attended_fresh_receive"]["fallback_draining_command"],
        )
        handoff = gates["attended_fresh_receive"]["operator_handoff"]
        self.assertEqual(handoff["preferred_profile"], "attended-cache-receive")
        self.assertEqual(handoff["fallback_profile"], "attended-send-receive")
        self.assertEqual(handoff["home_channel"], "20530681934008@lid")
        self.assertEqual(handoff["home_channel_kind"], "lid")
        self.assertEqual(handoff["agent_name"], "Quill")
        self.assertEqual(handoff["agent_number"], "13236478455")
        self.assertFalse(handoff["drains_bridge_messages"])
        self.assertTrue(handoff["fallback_drains_bridge_messages"])
        self.assertIn("audio_cache", handoff["audio_cache_dir"])
        self.assertEqual(len(handoff["steps"]), 3)
        self.assertEqual(
            gates["whatsapp_cloud_calling"]["status"],
            "external_setup_required",
        )
        cloud_gate = gates["whatsapp_cloud"]
        self.assertEqual(cloud_gate["status"], "external_setup_required")
        cloud_handoff = cloud_gate["setup_handoff"]
        self.assertEqual(
            cloud_handoff["required_keys"],
            [
                "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
                "WHATSAPP_CLOUD_ACCESS_TOKEN",
                "WHATSAPP_CLOUD_APP_SECRET",
                "WHATSAPP_CLOUD_VERIFY_TOKEN",
            ],
        )
        self.assertIn("WHATSAPP_CLOUD_ACCESS_TOKEN", cloud_handoff["missing"])
        self.assertIn("hermes/.env", cloud_handoff["env_file"])
        self.assertEqual(
            cloud_handoff["credential_sources"]["WHATSAPP_CLOUD_ACCESS_TOKEN"],
            [],
        )
        self.assertIn(
            "WHATSAPP_CLOUD_CALLING_SIDECAR_URL",
            cloud_handoff["available_source_keys"]["systemd_service"],
        )
        self.assertIn("--require-whatsapp-cloud", cloud_handoff["verify_command"])
        self.assertEqual(len(cloud_handoff["steps"]), 4)
        self.assertNotIn("phone-id", json.dumps(cloud_handoff))
        self.assertIn(
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            gates["whatsapp_cloud_calling"]["missing"],
        )
        calling_handoff = gates["whatsapp_cloud_calling"]["setup_handoff"]
        self.assertIn(
            "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND",
            calling_handoff["required_keys"],
        )
        self.assertEqual(
            calling_handoff["sidecar_sources"][
                "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND"
            ],
            ["systemd_service"],
        )
        self.assertEqual(calling_handoff["sidecar_missing"], [])
        self.assertTrue(calling_handoff["calling_sidecar_configured"])
        self.assertFalse(calling_handoff["calling_ready"])
        self.assertIn("--require-whatsapp-calling", calling_handoff["verify_command"])
        self.assertIn("--require-complete", calling_handoff["complete_verification_command"])
        self.assertEqual(len(calling_handoff["steps"]), 5)
        summary = payload["readiness_summary"]
        self.assertEqual(summary["status"], "local_ready_pending_gates")
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["local_checks_passed"])
        self.assertFalse(summary["attended_fresh_receive_verified"])
        self.assertTrue(summary["external_meta_setup_required"])
        self.assertTrue(summary["operator_action_required"])
        self.assertIn(
            "run_attended_fresh_receive",
            [action["id"] for action in summary["next_actions"]],
        )
        self.assertIn(
            "configure_whatsapp_cloud_calling",
            [action["id"] for action in summary["next_actions"]],
        )
        actions = {action["id"]: action for action in summary["next_actions"]}
        attended_action = actions["run_attended_fresh_receive"]
        self.assertEqual(
            attended_action["operator_handoff"]["preferred_profile"],
            "attended-cache-receive",
        )
        self.assertEqual(
            attended_action["operator_handoff"]["home_channel"],
            "20530681934008@lid",
        )
        meta_action = actions["configure_whatsapp_cloud_calling"]
        self.assertEqual(
            meta_action["gates"],
            ["whatsapp_cloud", "whatsapp_cloud_calling"],
        )
        self.assertIn(
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            meta_action["missing_by_gate"]["whatsapp_cloud"],
        )
        self.assertIn(
            "WHATSAPP_CLOUD_ACCESS_TOKEN",
            meta_action["missing_by_gate"]["whatsapp_cloud_calling"],
        )
        self.assertIn(
            "--require-whatsapp-cloud",
            meta_action["verify_commands"]["whatsapp_cloud"],
        )
        self.assertIn(
            "--require-whatsapp-calling",
            meta_action["verify_commands"]["whatsapp_cloud_calling"],
        )
        self.assertIn(
            "--require-complete",
            meta_action["complete_verification_command"],
        )

    def test_require_complete_fails_when_alpha_gates_are_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_readiness_process(
                Path(tmp),
                "--require-complete",
            )

        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["success"])
        self.assertEqual(
            payload["readiness_summary"]["status"],
            "local_ready_pending_gates",
        )
        failures = {
            item["name"]: item
            for item in payload["failures"]
        }
        self.assertIn("whatsapp_alpha_complete", failures)
        self.assertEqual(
            failures["whatsapp_alpha_complete"]["category"],
            "readiness_summary",
        )
        self.assertIn(
            "run_attended_fresh_receive",
            failures["whatsapp_alpha_complete"]["failures"][1],
        )

    def test_default_voice_bin_prefers_installed_voice_on_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            helpers = tmp_path / "helpers"
            bin_dir = tmp_path / "bin"
            helpers.mkdir()
            bin_dir.mkdir()
            write_fake_helpers(helpers)
            voice = bin_dir / "voice"
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config = tmp_path / "config.yaml"
            config.write_text("tts: {}\n", encoding="utf-8")

            env = {**os.environ}
            env.pop("VOICE_BIN", None)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["VOICE_READINESS_SCRIPT_DIR"] = str(helpers)

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--hermes-home",
                    str(tmp_path / "hermes"),
                    "--hermes-config",
                    str(config),
                    "--skip-systemd",
                    "--skip-daemon",
                    "--skip-sidecar",
                    "--skip-voice-note-smoke",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        expected = str(voice.resolve())
        for component in payload["components"]:
            command = component["command"]
            if "--voice-bin" in command:
                index = command.index("--voice-bin")
                self.assertEqual(command[index + 1], expected)

    def test_human_summary_prints_non_draining_attended_next_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            helpers = tmp_path / "helpers"
            helpers.mkdir()
            write_fake_helpers(helpers)
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
                ],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "VOICE_READINESS_SCRIPT_DIR": str(helpers),
                },
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "attended_fresh_receive_command=scripts/verify_whatsapp_alpha_readiness.py",
            result.stdout,
        )
        self.assertIn("--profile attended-cache-receive", result.stdout)
        self.assertIn(
            "attended_fresh_receive_fallback_draining_command=",
            result.stdout,
        )
        self.assertIn("--profile attended-send-receive", result.stdout)
        self.assertIn(
            "attended_fresh_receive_operator=agent=Quill number=13236478455",
            result.stdout,
        )
        self.assertIn(
            "home_channel=20530681934008@lid",
            result.stdout,
        )
        self.assertIn(
            "attended_fresh_receive_step[1]=Start the preferred command",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_cloud_setup=env_file=",
            result.stdout,
        )
        self.assertIn(
            "missing=WHATSAPP_CLOUD_PHONE_NUMBER_ID,WHATSAPP_CLOUD_ACCESS_TOKEN",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_cloud_verify_command=scripts/verify_whatsapp_alpha_readiness.py",
            result.stdout,
        )
        self.assertIn("--require-whatsapp-cloud", result.stdout)
        self.assertIn(
            "whatsapp_cloud_step[1]=Create or select the Meta app",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_calling_setup=sidecar_configured=True",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_calling_verify_command=scripts/verify_whatsapp_alpha_readiness.py",
            result.stdout,
        )
        self.assertIn("--require-whatsapp-calling", result.stdout)
        self.assertIn(
            "whatsapp_calling_complete_command=scripts/verify_whatsapp_alpha_readiness.py",
            result.stdout,
        )
        self.assertIn("--require-complete", result.stdout)
        self.assertIn(
            "whatsapp_calling_step[1]=Complete WhatsApp Cloud setup first",
            result.stdout,
        )
        self.assertIn("readiness=local_ready_pending_gates", result.stdout)

    def test_inbound_cache_smoke_adds_receive_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_cache = Path(tmp) / "audio_cache"
            payload = self.run_readiness(
                Path(tmp),
                "--run-inbound-cache-smoke",
                "--whatsapp-audio-cache-dir",
                str(audio_cache),
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
        self.assertEqual(
            payload["pending_gates"]["attended_fresh_receive"]["operator_handoff"][
                "audio_cache_dir"
            ],
            str(audio_cache.resolve()),
        )

    def test_cached_receive_profile_adds_receive_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_cache = Path(tmp) / "audio_cache"
            audio_cache.mkdir()
            cached_audio = audio_cache / "aud_cached.ogg"
            cached_audio.write_bytes(b"OggSfake")
            payload = self.run_readiness(
                Path(tmp),
                "--profile",
                "cached-receive",
                "--whatsapp-audio-cache-dir",
                str(audio_cache),
            )

        self.assertEqual(payload["profile"], "cached-receive")
        components = {item["name"]: item for item in payload["components"]}
        self.assertIn("whatsapp_inbound_cache_stt", components)
        hermes_command = components["hermes_voice_config"]["command"]
        self.assertIn("--stt-audio", hermes_command)
        self.assertIn(str(cached_audio.resolve()), hermes_command)
        self.assertIn("--stt-timeout", hermes_command)

    def test_cached_receive_profile_can_skip_hermes_stt_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_cache = Path(tmp) / "audio_cache"
            audio_cache.mkdir()
            (audio_cache / "aud_cached.ogg").write_bytes(b"OggSfake")
            payload = self.run_readiness(
                Path(tmp),
                "--profile",
                "cached-receive",
                "--whatsapp-audio-cache-dir",
                str(audio_cache),
                "--skip-hermes-stt-smoke",
            )

        components = {item["name"]: item for item in payload["components"]}
        self.assertNotIn("--stt-audio", components["hermes_voice_config"]["command"])

    def test_send_profile_posts_real_voice_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(
                Path(tmp),
                "--profile",
                "send",
                skip_voice_note_smoke=False,
            )

        self.assertEqual(payload["profile"], "send")
        components = {item["name"]: item for item in payload["components"]}
        self.assertIn("whatsapp_voice_note_send", components)
        self.assertIn("--send", components["whatsapp_voice_note_send"]["command"])

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

    def test_attended_send_receive_profile_expands_guarded_receive_flags(self):
        voice_note_payload = {
            "success": True,
            "checks": {
                "inbound_audio": {
                    "drains_bridge_messages": True,
                    "audio_events": [{"mediaType": "ptt"}],
                }
            },
            "failures": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(
                Path(tmp),
                "--profile",
                "attended-send-receive",
                skip_voice_note_smoke=False,
                voice_note_payload=voice_note_payload,
            )

        self.assertEqual(payload["profile"], "attended-send-receive")
        components = {item["name"]: item for item in payload["components"]}
        command = components["whatsapp_voice_note_send_receive"]["command"]
        self.assertIn("--send", command)
        self.assertIn("--wait-inbound-seconds", command)
        self.assertIn("60.0", command)
        self.assertIn("--require-inbound-audio", command)
        self.assertIn("--drain-bridge-messages", command)

    def test_attended_cache_receive_profile_watches_cache_without_draining_bridge(self):
        inbound_cache_payload = {
            "success": True,
            "checks": {
                "selected_files": ["/tmp/aud_fresh.ogg"],
                "audio": [{"path": "/tmp/aud_fresh.ogg"}],
                "fresh_watch": {
                    "drains_bridge_messages": False,
                    "fresh_files": ["/tmp/aud_fresh.ogg"],
                    "fresh_count": 1,
                }
            },
            "failures": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(
                Path(tmp),
                "--profile",
                "attended-cache-receive",
                skip_voice_note_smoke=False,
                inbound_cache_payload=inbound_cache_payload,
            )

        self.assertEqual(payload["profile"], "attended-cache-receive")
        components = {item["name"]: item for item in payload["components"]}
        self.assertIn("whatsapp_voice_note_send", components)
        self.assertIn("whatsapp_inbound_cache_fresh_stt", components)
        command = components["whatsapp_inbound_cache_fresh_stt"]["command"]
        self.assertIn("--wait-fresh-seconds", command)
        self.assertIn("60.0", command)
        self.assertIn("--require-fresh-audio", command)
        gate = payload["pending_gates"]["attended_fresh_receive"]
        self.assertEqual(gate["status"], "verified")
        self.assertEqual(gate["component"], "whatsapp_inbound_cache_fresh_stt")
        self.assertFalse(gate["drains_bridge_messages"])
        self.assertFalse(gate["requires_operator"])
        summary = payload["readiness_summary"]
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["attended_fresh_receive_verified"])
        self.assertTrue(summary["external_meta_setup_required"])
        self.assertEqual(
            [action["id"] for action in summary["next_actions"]],
            ["configure_whatsapp_cloud_calling"],
        )

    def test_cached_receive_verified_survives_optional_fresh_watch_timeout(self):
        inbound_cache_payload = {
            "success": True,
            "checks": {
                "selected_files": ["/tmp/aud_cached.ogg"],
                "audio": [{"path": "/tmp/aud_cached.ogg"}],
                "fresh_watch": {
                    "drains_bridge_messages": False,
                    "fresh_files": [],
                    "fresh_count": 0,
                },
            },
            "failures": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(
                Path(tmp),
                "--profile",
                "cached-receive",
                "--wait-audio-cache-seconds",
                "0.1",
                inbound_cache_payload=inbound_cache_payload,
            )

        gate = payload["pending_gates"]["attended_fresh_receive"]
        self.assertEqual(gate["component"], "whatsapp_inbound_cache_fresh_stt")
        self.assertEqual(gate["status"], "not_verified")
        self.assertTrue(gate["cached_receive_verified"])
        self.assertTrue(gate["requires_operator"])
        summary = payload["readiness_summary"]
        self.assertFalse(summary["attended_fresh_receive_verified"])
        self.assertIn(
            "run_attended_fresh_receive",
            [action["id"] for action in summary["next_actions"]],
        )

    def test_verified_attended_receive_gate_requires_audio_event_evidence(self):
        voice_note_payload = {
            "success": True,
            "checks": {
                "inbound_audio": {
                    "drains_bridge_messages": True,
                    "audio_events": [{"mediaType": "ptt"}],
                }
            },
            "failures": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.run_readiness(
                Path(tmp),
                "--send-voice-note",
                "--wait-inbound-seconds",
                "5",
                "--require-inbound-audio",
                "--drain-bridge-messages",
                skip_voice_note_smoke=False,
                voice_note_payload=voice_note_payload,
            )

        gate = payload["pending_gates"]["attended_fresh_receive"]
        self.assertEqual(gate["status"], "verified")
        self.assertEqual(gate["component"], "whatsapp_voice_note_send_receive")
        self.assertEqual(gate["audio_events"], 1)
        self.assertFalse(gate["requires_operator"])

    def test_readiness_summary_is_complete_when_all_gates_are_verified(self):
        inbound_cache_payload = {
            "success": True,
            "checks": {
                "fresh_watch": {
                    "drains_bridge_messages": False,
                    "fresh_files": ["/tmp/aud_fresh.ogg"],
                    "fresh_count": 1,
                }
            },
            "failures": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            helpers = tmp_path / "helpers"
            helpers.mkdir()
            write_fake_helpers(
                helpers,
                cloud_configured=True,
                inbound_cache_payload=inbound_cache_payload,
            )
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
                    "--profile",
                    "attended-cache-receive",
                    "--require-complete",
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

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        summary = payload["readiness_summary"]
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["local_checks_passed"])
        self.assertTrue(summary["attended_fresh_receive_verified"])
        self.assertFalse(summary["external_meta_setup_required"])
        self.assertFalse(summary["operator_action_required"])
        self.assertEqual(summary["next_actions"], [])
        gates = payload["pending_gates"]
        self.assertEqual(gates["whatsapp_cloud"]["status"], "configured")
        self.assertEqual(gates["whatsapp_cloud"]["setup_handoff"]["steps"], [])
        self.assertEqual(gates["whatsapp_cloud"]["setup_handoff"]["missing"], [])
        self.assertEqual(gates["whatsapp_cloud_calling"]["status"], "ready")
        self.assertEqual(
            gates["whatsapp_cloud_calling"]["setup_handoff"]["steps"],
            [],
        )
        self.assertEqual(
            gates["whatsapp_cloud_calling"]["setup_handoff"]["missing"],
            [],
        )

    def test_voice_note_flags_cannot_be_used_when_voice_note_smoke_is_skipped(self):
        result = self.run_invalid("--skip-voice-note-smoke", "--send-voice-note")

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be used with --skip-voice-note-smoke", result.stderr)

    def test_real_voice_note_profiles_cannot_skip_voice_note_smoke(self):
        result = self.run_invalid(
            "--profile",
            "attended-send-receive",
            "--skip-voice-note-smoke",
        )

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

    def test_require_fresh_cache_audio_requires_wait_window(self):
        result = self.run_invalid("--require-fresh-cache-audio")

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--require-fresh-cache-audio requires --wait-audio-cache-seconds",
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
