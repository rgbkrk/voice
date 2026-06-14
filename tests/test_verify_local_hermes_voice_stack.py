#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_local_hermes_voice_stack.sh"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def write_helper(path: Path, label: str, log_path: Path) -> None:
    write_executable(
        path,
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '{label}' >> {str(log_path)!r}
            printf '\\0' >> {str(log_path)!r}
            printf '%s\\0' "$@" >> {str(log_path)!r}
            printf '\\n' >> {str(log_path)!r}
            echo "ok: {label}"
            """
        ),
    )


def write_fake_python(path: Path, label: str, log_path: Path) -> None:
    write_executable(
        path,
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '{label}' >> {str(log_path)!r}
            printf '\\0' >> {str(log_path)!r}
            printf '%s\\0' "$@" >> {str(log_path)!r}
            printf '\\n' >> {str(log_path)!r}
            echo '{{"success": true}}'
            """
        ),
    )


def command_log_entries(log_path: Path) -> list[list[str]]:
    entries: list[list[str]] = []
    for line in log_path.read_bytes().splitlines():
        if not line:
            continue
        entries.append(line.decode("utf-8").split("\0")[:-1])
    return entries


class LocalHermesVoiceStackVerifierTests(unittest.TestCase):
    def test_default_gate_runs_strict_release_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_helper(hermes, "hermes", log_path)
            write_helper(gateway, "gateway", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(bridge, "bridge", log_path)
            write_helper(sidecar, "sidecar", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(config),
                    "--sidecar-url",
                    "http://127.0.0.1:9999",
                    "--skip-systemd",
                    "--text",
                    "Stack smoke.",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("ok: local Hermes voice stack verifier passed", result.stdout)
        self.assertIn("hermes_gateway=skipped", result.stdout)
        self.assertIn("cli_mcp=checked", result.stdout)
        self.assertIn("whatsapp_bridge=checked", result.stdout)
        self.assertIn("whatsapp_inbound_cache=skipped", result.stdout)
        self.assertIn("whatsapp_alpha=skipped", result.stdout)
        self.assertIn("webrtc_loopback=skipped", result.stdout)
        self.assertEqual(
            [entry[0] for entry in entries],
            ["hermes", "cli_mcp", "whatsapp", "bridge", "sidecar"],
        )
        self.assertEqual(
            entries[0],
            [
                "hermes",
                "--config",
                str(config),
                "--voice-bin",
                str(voice),
                "--text",
                "Stack smoke.",
            ],
        )
        self.assertEqual(
            entries[1],
            [
                "cli_mcp",
                "--voice-bin",
                str(voice),
                "--require-daemon",
            ],
        )
        self.assertEqual(
            entries[2],
            [
                "whatsapp",
                "--voice-bin",
                str(voice),
                "--text",
                "Stack smoke.",
                "--require-daemon",
                "--run-stt-smoke",
            ],
        )
        self.assertEqual(
            entries[3],
            [
                "bridge",
                "--hermes-home",
                str(Path.home() / ".hermes"),
                "--bridge-url",
                "http://127.0.0.1:3000",
                "--skip-systemd",
            ],
        )
        self.assertEqual(
            entries[4],
            [
                "sidecar",
                "--voice-bin",
                str(voice),
                "--sidecar-url",
                "http://127.0.0.1:9999",
                "--skip-systemd",
            ],
        )

    def test_skip_flags_disable_optional_release_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_helper(hermes, "hermes", log_path)
            write_helper(gateway, "gateway", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(sidecar, "sidecar", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(config),
                    "--skip-hermes-tts-smoke",
                    "--skip-hermes-gateway",
                    "--skip-sidecar",
                    "--skip-whatsapp-bridge",
                    "--skip-daemon",
                    "--skip-stt-smoke",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("sidecar_service=skipped", result.stdout)
        self.assertIn("hermes_gateway=skipped", result.stdout)
        self.assertIn("whatsapp_bridge=skipped", result.stdout)
        self.assertIn("whatsapp_inbound_cache=skipped", result.stdout)
        self.assertIn("whatsapp_alpha=skipped", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["hermes", "cli_mcp", "whatsapp"])
        self.assertIn("--skip-tts-smoke", entries[0])
        self.assertIn("--skip-daemon", entries[1])
        self.assertIn("--skip-daemon", entries[2])
        self.assertNotIn("--run-stt-smoke", entries[2])

    def test_webrtc_loopback_smoke_runs_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            sidecar = tmp_path / "verify_sidecar.py"
            python = tmp_path / "python"
            smoke = tmp_path / "full_duplex_loopback_smoke.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_helper(hermes, "hermes", log_path)
            write_helper(gateway, "gateway", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(bridge, "bridge", log_path)
            write_helper(sidecar, "sidecar", log_path)
            write_fake_python(python, "webrtc", log_path)
            write_executable(smoke, "#!/usr/bin/env python3\n")
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(config),
                    "--skip-hermes-tts-smoke",
                    "--skip-sidecar",
                    "--skip-daemon",
                    "--run-webrtc-loopback-smoke",
                    "--webrtc-python",
                    str(python),
                    "--webrtc-timeout",
                    "12.5",
                    "--max-queued-tx-ms",
                    "250",
                    "--text",
                    "Outbound media smoke.",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                    "WEBRTC_LOOPBACK_SMOKE_SCRIPT": str(smoke),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("webrtc_loopback=checked", result.stdout)
        self.assertEqual(
            [entry[0] for entry in entries],
            ["hermes", "gateway", "cli_mcp", "whatsapp", "bridge", "webrtc"],
        )
        self.assertEqual(
            entries[5],
            [
                "webrtc",
                str(smoke),
                "--voice-bin",
                str(voice),
                "--sidecar-url",
                "http://127.0.0.1:8787",
                "--timeout",
                "12.5",
                "--outbound-text",
                "Outbound media smoke.",
                "--max-queued-tx-ms",
                "250",
            ],
        )

    def test_whatsapp_inbound_cache_smoke_runs_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            inbound = tmp_path / "verify_inbound_cache.py"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            audio_cache = tmp_path / "audio_cache"

            write_helper(hermes, "hermes", log_path)
            write_helper(gateway, "gateway", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(bridge, "bridge", log_path)
            write_helper(inbound, "inbound", log_path)
            write_helper(sidecar, "sidecar", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            audio_cache.mkdir()

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(tmp_path / "missing.yaml"),
                    "--hermes-home",
                    str(tmp_path / "hermes"),
                    "--skip-hermes-config",
                    "--skip-hermes-gateway",
                    "--skip-cli-mcp",
                    "--skip-sidecar",
                    "--skip-daemon",
                    "--skip-whatsapp-bridge",
                    "--run-whatsapp-inbound-cache-smoke",
                    "--whatsapp-audio-cache-dir",
                    str(audio_cache),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "WHATSAPP_INBOUND_CACHE_VERIFY_SCRIPT": str(inbound),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("whatsapp_inbound_cache=checked", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "inbound"])
        self.assertEqual(
            entries[1],
            [
                "inbound",
                "--voice-bin",
                str(voice),
                "--hermes-home",
                str(tmp_path / "hermes"),
                "--require-cache",
                "--run-stt",
                "--audio-cache-dir",
                str(audio_cache),
            ],
        )

    def test_whatsapp_alpha_profile_runs_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            alpha = tmp_path / "verify_alpha.py"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"
            hermes_home = tmp_path / "hermes"
            audio_cache = tmp_path / "audio_cache"

            write_helper(hermes, "hermes", log_path)
            write_helper(gateway, "gateway", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(bridge, "bridge", log_path)
            write_helper(alpha, "alpha", log_path)
            write_helper(sidecar, "sidecar", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")
            hermes_home.mkdir()
            audio_cache.mkdir()

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(config),
                    "--hermes-home",
                    str(hermes_home),
                    "--sidecar-url",
                    "http://127.0.0.1:9999",
                    "--whatsapp-bridge-url",
                    "http://127.0.0.1:3001",
                    "--whatsapp-audio-cache-dir",
                    str(audio_cache),
                    "--expected-whatsapp-agent-number",
                    "13236478455",
                    "--expected-whatsapp-agent-name",
                    "Quill",
                    "--require-whatsapp-cloud",
                    "--require-whatsapp-calling",
                    "--require-whatsapp-alpha-complete",
                    "--check-whatsapp-cloud-api",
                    "--skip-hermes-config",
                    "--skip-hermes-gateway",
                    "--skip-cli-mcp",
                    "--skip-sidecar",
                    "--skip-whatsapp-bridge",
                    "--skip-systemd",
                    "--skip-daemon",
                    "--skip-hermes-tts-smoke",
                    "--skip-hermes-stt-smoke",
                    "--whatsapp-alpha-profile",
                    "attended-cache-receive",
                    "--whatsapp-alpha-chat-id",
                    "20530681934008@lid",
                    "--whatsapp-alpha-wait-audio-cache-seconds",
                    "7.5",
                    "--whatsapp-alpha-wait-inbound-seconds",
                    "8.5",
                    "--text",
                    "Alpha stack smoke.",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("whatsapp_alpha=attended-cache-receive", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "alpha"])
        self.assertEqual(
            entries[1],
            [
                "alpha",
                "--voice-bin",
                str(voice),
                "--hermes-home",
                str(hermes_home),
                "--hermes-config",
                str(config),
                "--bridge-url",
                "http://127.0.0.1:3001",
                "--sidecar-url",
                "http://127.0.0.1:9999",
                "--profile",
                "attended-cache-receive",
                "--text",
                "Alpha stack smoke.",
                "--whatsapp-audio-cache-dir",
                str(audio_cache),
                "--voice-note-chat-id",
                "20530681934008@lid",
                "--wait-audio-cache-seconds",
                "7.5",
                "--wait-inbound-seconds",
                "8.5",
                "--expected-agent-number",
                "13236478455",
                "--expected-agent-name",
                "Quill",
                "--skip-systemd",
                "--skip-daemon",
                "--skip-sidecar",
                "--skip-hermes-tts-smoke",
                "--skip-hermes-stt-smoke",
                "--require-whatsapp-cloud",
                "--require-whatsapp-calling",
                "--check-whatsapp-cloud-api",
                "--require-complete",
            ],
        )

    def test_whatsapp_alpha_text_overrides_only_alpha_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            alpha = tmp_path / "verify_alpha.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_helper(hermes, "hermes", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(alpha, "alpha", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(config),
                    "--skip-hermes-gateway",
                    "--skip-whatsapp-bridge",
                    "--skip-sidecar",
                    "--skip-systemd",
                    "--skip-daemon",
                    "--whatsapp-alpha-profile",
                    "attended-cache-receive",
                    "--whatsapp-alpha-text",
                    "Please send a short test voice note back to Quill.",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("whatsapp_alpha=attended-cache-receive", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["hermes", "cli_mcp", "whatsapp", "alpha"])
        self.assertIn("--text", entries[0])
        self.assertIn("Local Hermes voice stack smoke test.", entries[0])
        alpha_entry = entries[3]
        self.assertIn("--text", alpha_entry)
        self.assertIn("Please send a short test voice note back to Quill.", alpha_entry)
        self.assertNotIn("Local Hermes voice stack smoke test.", alpha_entry)

    def test_require_whatsapp_alpha_complete_requires_alpha_profile(self):
        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--require-whatsapp-alpha-complete",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--require-whatsapp-alpha-complete requires --whatsapp-alpha-profile",
            result.stderr,
        )

    def test_whatsapp_alpha_json_output_requires_alpha_profile(self):
        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--whatsapp-alpha-json-output",
                "alpha.json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--whatsapp-alpha-json-output requires --whatsapp-alpha-profile",
            result.stderr,
        )

    def test_whatsapp_alpha_json_output_saves_alpha_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            alpha = tmp_path / "verify_alpha.py"
            voice = tmp_path / "voice"
            json_output = tmp_path / "reports" / "alpha.json"

            write_helper(whatsapp, "whatsapp", log_path)
            write_executable(
                alpha,
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf 'alpha' >> {str(log_path)!r}
                    printf '\\0' >> {str(log_path)!r}
                    printf '%s\\0' "$@" >> {str(log_path)!r}
                    printf '\\n' >> {str(log_path)!r}
                    cat <<'JSON'
                    {{
                      "profile": "cached-receive",
                      "readiness_summary": {{
                        "status": "local_ready_pending_gates",
                        "complete": false,
                        "local_checks_passed": true,
                        "attended_fresh_receive_verified": true,
                        "external_meta_setup_required": true,
                        "operator_action_required": true,
                        "next_actions": [
                          {{"id": "configure_whatsapp_cloud_calling"}}
                        ]
                      }},
                      "pending_gates": {{
                        "attended_fresh_receive": {{
                          "status": "verified",
                          "cached_receive_verified": true,
                          "evidence": {{
                            "kind": "audio_cache",
                            "fresh": true,
                            "fresh_count": 1,
                            "drains_bridge_messages": false,
                            "audio": [
                              {{
                                "codec": "opus",
                                "stt": {{
                                  "frames": 210,
                                  "audio_duration_ms": 4200,
                                  "tokens": 6,
                                  "text_redacted": true,
                                  "text_chars": 42
                                }}
                              }}
                            ]
                          }}
                        }},
                        "whatsapp_cloud": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": [
                              "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
                              "WHATSAPP_CLOUD_ACCESS_TOKEN"
                            ],
                            "invalid": [],
                            "verify_command": [
                              "scripts/verify_whatsapp_alpha_readiness.py",
                              "--hermes-home",
                              "/home/ubuntu/.hermes",
                              "--require-whatsapp-cloud",
                              "--check-whatsapp-cloud-api"
                            ]
                          }}
                        }},
                        "whatsapp_cloud_calling": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": [
                              "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
                              "WHATSAPP_CLOUD_ACCESS_TOKEN",
                              "WHATSAPP_CLOUD_APP_SECRET",
                              "WHATSAPP_CLOUD_VERIFY_TOKEN"
                            ],
                            "invalid": [],
                            "verify_command": [
                              "scripts/verify_whatsapp_alpha_readiness.py",
                              "--hermes-home",
                              "/home/ubuntu/.hermes",
                              "--require-whatsapp-cloud",
                              "--require-whatsapp-calling",
                              "--check-whatsapp-cloud-api"
                            ],
                            "complete_verification_command": [
                              "scripts/verify_whatsapp_alpha_readiness.py",
                              "--hermes-home",
                              "/home/ubuntu/.hermes",
                              "--profile",
                              "attended-cache-receive",
                              "--require-complete"
                            ]
                          }}
                        }}
                      }}
                    }}
                    JSON
                    """
                ),
            )
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(tmp_path / "missing.yaml"),
                    "--skip-hermes-config",
                    "--skip-hermes-gateway",
                    "--skip-cli-mcp",
                    "--skip-sidecar",
                    "--skip-whatsapp-bridge",
                    "--skip-systemd",
                    "--skip-daemon",
                    "--whatsapp-alpha-profile",
                    "cached-receive",
                    "--whatsapp-alpha-json-output",
                    str(json_output),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )

            entries = command_log_entries(log_path)
            saved_output = json_output.read_text(encoding="utf-8").strip()
            json_output_exists = json_output.is_file()

        self.assertTrue(json_output_exists)
        self.assertIn('"profile": "cached-receive"', saved_output)
        self.assertIn(f"whatsapp_alpha_json={json_output}", result.stdout)
        self.assertIn("whatsapp_alpha_json_profile=cached-receive", result.stdout)
        self.assertIn(
            "whatsapp_alpha_json_readiness=local_ready_pending_gates complete=False "
            "local_checks_passed=True attended_fresh_receive_verified=True "
            "external_meta_setup_required=True operator_action_required=True",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_next_actions=configure_whatsapp_cloud_calling",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_attended_fresh_receive=verified "
            "cached_receive_verified=True",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_attended_evidence=kind=audio_cache fresh=True "
            "drains_messages=False audio_events=1 codec=opus text_chars=42",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_cloud=external_setup_required "
            "missing=WHATSAPP_CLOUD_PHONE_NUMBER_ID,WHATSAPP_CLOUD_ACCESS_TOKEN "
            "invalid=none",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_cloud_verify_command="
            "scripts/verify_whatsapp_alpha_readiness.py --hermes-home "
            "/home/ubuntu/.hermes --require-whatsapp-cloud --check-whatsapp-cloud-api",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_calling=external_setup_required "
            "missing=WHATSAPP_CLOUD_PHONE_NUMBER_ID,WHATSAPP_CLOUD_ACCESS_TOKEN,"
            "WHATSAPP_CLOUD_APP_SECRET,WHATSAPP_CLOUD_VERIFY_TOKEN invalid=none",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_calling_verify_command="
            "scripts/verify_whatsapp_alpha_readiness.py --hermes-home "
            "/home/ubuntu/.hermes --require-whatsapp-cloud "
            "--require-whatsapp-calling --check-whatsapp-cloud-api",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_calling_complete_command="
            "scripts/verify_whatsapp_alpha_readiness.py --hermes-home "
            "/home/ubuntu/.hermes --profile attended-cache-receive "
            "--require-complete",
            result.stdout,
        )
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "alpha"])
        self.assertIn("--json", entries[1])

    def test_whatsapp_alpha_json_summary_prints_attended_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            alpha = tmp_path / "verify_alpha.py"
            voice = tmp_path / "voice"
            json_output = tmp_path / "alpha.json"

            write_helper(whatsapp, "whatsapp", log_path)
            write_executable(
                alpha,
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf 'alpha' >> {str(log_path)!r}
                    printf '\\0' >> {str(log_path)!r}
                    printf '%s\\0' "$@" >> {str(log_path)!r}
                    printf '\\n' >> {str(log_path)!r}
                    cat <<'JSON'
                    {{
                      "profile": "cached-receive",
                      "readiness_summary": {{
                        "status": "local_ready_pending_gates",
                        "complete": false,
                        "local_checks_passed": true,
                        "attended_fresh_receive_verified": false,
                        "external_meta_setup_required": false,
                        "operator_action_required": true,
                        "next_actions": [
                          {{"id": "run_attended_fresh_receive"}}
                        ]
                      }},
                      "pending_gates": {{
                        "attended_fresh_receive": {{
                          "status": "not_verified",
                          "cached_receive_verified": true,
                          "command": [
                            "scripts/verify_whatsapp_alpha_readiness.py",
                            "--hermes-home",
                            "/home/ubuntu/.hermes",
                            "--profile",
                            "attended-cache-receive",
                            "--wait-audio-cache-seconds",
                            "60.0"
                          ],
                          "fallback_draining_command": [
                            "scripts/verify_whatsapp_alpha_readiness.py",
                            "--hermes-home",
                            "/home/ubuntu/.hermes",
                            "--profile",
                            "attended-send-receive",
                            "--wait-inbound-seconds",
                            "60.0"
                          ]
                        }},
                        "whatsapp_cloud": {{"status": "configured"}},
                        "whatsapp_cloud_calling": {{"status": "ready"}}
                      }}
                    }}
                    JSON
                    """
                ),
            )
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--skip-hermes-config",
                    "--skip-hermes-gateway",
                    "--skip-cli-mcp",
                    "--skip-sidecar",
                    "--skip-whatsapp-bridge",
                    "--skip-systemd",
                    "--skip-daemon",
                    "--whatsapp-alpha-profile",
                    "cached-receive",
                    "--whatsapp-alpha-json-output",
                    str(json_output),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )

        self.assertIn(
            "whatsapp_alpha_json_attended_command="
            "scripts/verify_whatsapp_alpha_readiness.py --hermes-home "
            "/home/ubuntu/.hermes --profile attended-cache-receive "
            "--wait-audio-cache-seconds 60.0",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_attended_fallback_draining_command="
            "scripts/verify_whatsapp_alpha_readiness.py --hermes-home "
            "/home/ubuntu/.hermes --profile attended-send-receive "
            "--wait-inbound-seconds 60.0",
            result.stdout,
        )

    def test_skip_hermes_config_does_not_require_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"

            write_helper(hermes, "hermes", log_path)
            write_helper(gateway, "gateway", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(bridge, "bridge", log_path)
            write_helper(sidecar, "sidecar", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(tmp_path / "missing.yaml"),
                    "--skip-hermes-config",
                    "--skip-cli-mcp",
                    "--skip-sidecar",
                    "--skip-daemon",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("hermes_config=skipped", result.stdout)
        self.assertIn("hermes_gateway=checked", result.stdout)
        self.assertIn("cli_mcp=skipped", result.stdout)
        self.assertIn("whatsapp_bridge=checked", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["gateway", "whatsapp", "bridge"])

    def test_gateway_service_check_runs_when_systemd_checks_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"
            hermes_home = tmp_path / "hermes"

            write_helper(hermes, "hermes", log_path)
            write_helper(gateway, "gateway", log_path)
            write_helper(cli_mcp, "cli_mcp", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(bridge, "bridge", log_path)
            write_helper(sidecar, "sidecar", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")
            hermes_home.mkdir()

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(config),
                    "--hermes-home",
                    str(hermes_home),
                    "--sidecar-url",
                    "http://127.0.0.1:9999",
                    "--skip-daemon",
                    "--text",
                    "Stack smoke.",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("hermes_gateway=checked", result.stdout)
        self.assertEqual(
            [entry[0] for entry in entries],
            ["hermes", "gateway", "cli_mcp", "whatsapp", "bridge", "sidecar"],
        )
        self.assertEqual(
            entries[1],
            [
                "gateway",
                "--voice-bin",
                str(voice),
                "--hermes-home",
                str(hermes_home),
                "--sidecar-url",
                "http://127.0.0.1:9999",
            ],
        )


if __name__ == "__main__":
    unittest.main()
