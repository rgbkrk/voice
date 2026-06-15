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


def write_failing_helper(path: Path, label: str, log_path: Path, status: int = 17) -> None:
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
            echo "failing {label}" >&2
            exit {status}
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


def write_external_meta_bridge_helper(path: Path, label: str, log_path: Path) -> None:
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
            cat <<'JSON'
            {{
              "success": false,
              "checks": {{
                "baileys_identity": {{
                  "name": "Quill",
                  "number": "13236478455",
                  "lid_number": "186999436771390"
                }},
                "bridge_health": {{
                  "status": "connected",
                  "queueLength": 0
                }},
                "whatsapp_cloud": {{
                  "cloud_configured": false,
                  "calling_ready": false,
                  "calling_sidecar_configured": true,
                  "cloud_missing": [
                    "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
                    "WHATSAPP_CLOUD_ACCESS_TOKEN",
                    "WHATSAPP_CLOUD_APP_SECRET",
                    "WHATSAPP_CLOUD_VERIFY_TOKEN"
                  ],
                  "cloud_invalid": [],
                  "calling_missing": [
                    "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
                    "WHATSAPP_CLOUD_ACCESS_TOKEN",
                    "WHATSAPP_CLOUD_APP_SECRET",
                    "WHATSAPP_CLOUD_VERIFY_TOKEN"
                  ],
                  "calling_invalid": [],
                  "webhook": {{
                    "host": "127.0.0.1",
                    "port": "8090",
                    "path": "/webhook",
                    "api_version": "v23.0",
                    "defaulted": ["WHATSAPP_CLOUD_WEBHOOK_HOST"],
                    "invalid": []
                  }},
                  "cloud_health": {{
                    "checked": true,
                    "ok": false,
                    "local_url": "http://127.0.0.1:8090/health",
                    "missing": ["WHATSAPP_CLOUD_PHONE_NUMBER_ID"],
                    "invalid": [],
                    "error": {{
                      "message": "Cloud health check requires a local health URL and phone number ID"
                    }}
                  }},
                  "cloud_api": {{"checked": false}},
                  "webhook_challenge": {{"checked": false}}
                }}
              }},
              "failures": [
                "WhatsApp Cloud health check failed: Cloud health check requires a local health URL and phone number ID"
              ]
            }}
            JSON
            exit 1
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
            install = tmp_path / "install_hermes.py"
            gateway = tmp_path / "verify_gateway.py"
            cli_mcp = tmp_path / "verify_cli_mcp.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_helper(hermes, "hermes", log_path)
            write_helper(install, "install", log_path)
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
                    "HERMES_CONFIG_INSTALL_SCRIPT": str(install),
                    "HERMES_GATEWAY_VERIFY_SCRIPT": str(gateway),
                    "CLI_MCP_SURFACE_VERIFY_SCRIPT": str(cli_mcp),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("ok: local Hermes voice stack verifier passed", result.stdout)
        self.assertIn("hermes_config_install=dry_run", result.stdout)
        self.assertIn("hermes_gateway=skipped", result.stdout)
        self.assertIn("cli_mcp=checked", result.stdout)
        self.assertIn("whatsapp_bridge=checked", result.stdout)
        self.assertIn("telegram_voice_contract=skipped", result.stdout)
        self.assertIn("whatsapp_inbound_cache=skipped", result.stdout)
        self.assertIn("whatsapp_alpha=skipped", result.stdout)
        self.assertIn("whatsapp_attended_watch=skipped", result.stdout)
        self.assertIn("webrtc_loopback=skipped", result.stdout)
        self.assertEqual(
            [entry[0] for entry in entries],
            ["hermes", "install", "cli_mcp", "whatsapp", "bridge", "sidecar"],
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
                "install",
                "--config",
                str(config),
                "--voice-bin",
                str(voice),
            ],
        )
        self.assertEqual(
            entries[2],
            [
                "cli_mcp",
                "--voice-bin",
                str(voice),
                "--require-daemon",
            ],
        )
        self.assertEqual(
            entries[3],
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
            entries[4],
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
            entries[5],
            [
                "sidecar",
                "--voice-bin",
                str(voice),
                "--sidecar-url",
                "http://127.0.0.1:9999",
                "--skip-systemd",
            ],
        )

    def test_attended_watch_status_runs_when_systemd_checks_are_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            watch = tmp_path / "start_watch.py"
            voice = tmp_path / "voice"
            output_dir = tmp_path / "watch-artifacts"

            write_helper(whatsapp, "whatsapp", log_path)
            write_executable(
                watch,
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf 'watch' >> {str(log_path)!r}
                    printf '\\0' >> {str(log_path)!r}
                    printf '%s\\0' "$@" >> {str(log_path)!r}
                    printf '\\n' >> {str(log_path)!r}
                    echo "ok: WhatsApp attended cache watch list"
                    echo "count=1"
                    echo "watch[1]=watch-active status=waiting_for_fresh_audio active_state=active"
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
                    "--skip-daemon",
                    "--whatsapp-attended-watch-output-dir",
                    str(output_dir),
                    "--whatsapp-attended-watch-unit-prefix",
                    "watch",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_ATTENDED_WATCH_SCRIPT": str(watch),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("==> WhatsApp attended cache watch status", result.stdout)
        self.assertIn("ok: WhatsApp attended cache watch list", result.stdout)
        self.assertIn("whatsapp_attended_watch=checked", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "watch", "watch"])
        self.assertEqual(
            entries[1],
            [
                "watch",
                "--list",
                "--output-dir",
                str(output_dir),
                "--unit-prefix",
                "watch",
            ],
        )
        self.assertEqual(entries[2][0], "watch")
        self.assertIn("--json", entries[2])

    def test_step_failure_reports_hermes_config_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            install = tmp_path / "install_hermes.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_failing_helper(hermes, "hermes", log_path, status=17)
            write_helper(install, "install", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
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
                    "--skip-cli-mcp",
                    "--skip-sidecar",
                    "--skip-whatsapp-bridge",
                    "--skip-daemon",
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "HERMES_CONFIG_INSTALL_SCRIPT": str(install),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                },
            )

            entries = command_log_entries(log_path)

        self.assertEqual(result.returncode, 17)
        self.assertEqual([entry[0] for entry in entries], ["hermes"])
        self.assertIn("error: local Hermes voice stack step failed: Hermes voice-native config", result.stderr)
        self.assertIn("failure_category=hermes_config", result.stderr)
        self.assertIn("failure_step=Hermes voice-native config", result.stderr)
        self.assertIn("failure_status=17", result.stderr)
        self.assertNotIn("ok: local Hermes voice stack verifier passed", result.stdout)

    def test_step_failure_reports_bridge_or_credentials_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            voice = tmp_path / "voice"

            write_helper(whatsapp, "whatsapp", log_path)
            write_failing_helper(bridge, "bridge", log_path, status=23)
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
                    "--skip-daemon",
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                },
            )

            entries = command_log_entries(log_path)

        self.assertEqual(result.returncode, 23)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "bridge"])
        self.assertIn(
            "error: local Hermes voice stack step failed: "
            "WhatsApp bridge identity and credential readiness",
            result.stderr,
        )
        self.assertIn("failure_category=whatsapp_bridge_or_credentials", result.stderr)
        self.assertIn("failure_step=WhatsApp bridge identity and credential readiness", result.stderr)
        self.assertIn("failure_status=23", result.stderr)
        self.assertNotIn("ok: local Hermes voice stack verifier passed", result.stdout)

    def test_cloud_probe_bridge_failure_continues_to_alpha_as_external_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            bridge = tmp_path / "verify_whatsapp_bridge.py"
            alpha = tmp_path / "verify_alpha.py"
            voice = tmp_path / "voice"

            write_helper(whatsapp, "whatsapp", log_path)
            write_external_meta_bridge_helper(bridge, "bridge", log_path)
            write_helper(alpha, "alpha", log_path)
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
                    "--skip-systemd",
                    "--skip-daemon",
                    "--check-whatsapp-cloud-health",
                    "--whatsapp-alpha-profile",
                    "cached-receive",
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_BRIDGE_VERIFY_SCRIPT": str(bridge),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )

            entries = command_log_entries(log_path)

        self.assertEqual(result.returncode, 1)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "bridge", "alpha"])
        self.assertIn("--json", entries[1])
        self.assertIn("--check-whatsapp-cloud-health", entries[1])
        self.assertIn(
            "whatsapp_bridge_json_cloud_health=failed",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_bridge_json_calling=not_ready sidecar_configured=True "
            "missing=WHATSAPP_CLOUD_PHONE_NUMBER_ID,WHATSAPP_CLOUD_ACCESS_TOKEN,"
            "WHATSAPP_CLOUD_APP_SECRET,WHATSAPP_CLOUD_VERIFY_TOKEN invalid=none",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_bridge_json_webhook=host=127.0.0.1 port=8090 "
            "path=/webhook api_version=v23.0 "
            "defaulted=WHATSAPP_CLOUD_WEBHOOK_HOST invalid=none",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_bridge_json_failure=WhatsApp Cloud health check failed",
            result.stdout,
        )
        self.assertIn("whatsapp_alpha=cached-receive", result.stdout)
        self.assertIn(
            "whatsapp_bridge=external_meta_setup_pending",
            result.stdout,
        )
        self.assertIn(
            "error: local Hermes voice stack external Meta setup check failed",
            result.stderr,
        )
        self.assertIn("failure_category=external_meta_setup", result.stderr)
        self.assertNotIn(
            "failure_category=whatsapp_bridge_or_credentials",
            result.stderr,
        )
        self.assertNotIn("ok: local Hermes voice stack verifier passed", result.stdout)

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
                    "--skip-hermes-install-dry-run",
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
        self.assertIn("hermes_config_install=skipped", result.stdout)
        self.assertIn("hermes_gateway=skipped", result.stdout)
        self.assertIn("whatsapp_bridge=skipped", result.stdout)
        self.assertIn("telegram_voice_contract=skipped", result.stdout)
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
                    "--skip-hermes-install-dry-run",
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

    def test_telegram_voice_contract_runs_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            telegram = tmp_path / "verify_telegram.sh"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"
            env_file = tmp_path / ".env"

            write_helper(whatsapp, "whatsapp", log_path)
            write_helper(telegram, "telegram", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")
            env_file.write_text("TELEGRAM_BOT_TOKEN=123:abc\n", encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-config",
                    str(config),
                    "--skip-hermes-config",
                    "--skip-hermes-gateway",
                    "--skip-cli-mcp",
                    "--skip-sidecar",
                    "--skip-daemon",
                    "--skip-whatsapp-bridge",
                    "--run-telegram-voice-contract",
                    "--telegram-env-file",
                    str(env_file),
                    "--require-telegram-credentials",
                    "--text",
                    "Telegram stack smoke.",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "TELEGRAM_CONTRACT_VERIFY_SCRIPT": str(telegram),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("telegram_voice_contract=checked", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "telegram"])
        self.assertEqual(
            entries[1],
            [
                "telegram",
                "--voice-bin",
                str(voice),
                "--text",
                "Telegram stack smoke.",
                "--hermes-config",
                str(config),
                "--hermes-env",
                str(env_file),
                "--skip-hermes-config",
                "--require-telegram-credentials",
                "--skip-daemon",
            ],
        )

    def test_step_failure_reports_telegram_setup_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            telegram = tmp_path / "verify_telegram.sh"
            voice = tmp_path / "voice"

            write_helper(whatsapp, "whatsapp", log_path)
            write_failing_helper(telegram, "telegram", log_path, status=31)
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
                    "--skip-daemon",
                    "--skip-whatsapp-bridge",
                    "--run-telegram-voice-contract",
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "TELEGRAM_CONTRACT_VERIFY_SCRIPT": str(telegram),
                },
            )

            entries = command_log_entries(log_path)

        self.assertEqual(result.returncode, 31)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "telegram"])
        self.assertIn(
            "error: local Hermes voice stack step failed: Telegram voice contract",
            result.stderr,
        )
        self.assertIn("failure_category=telegram_setup", result.stderr)
        self.assertIn("failure_step=Telegram voice contract", result.stderr)
        self.assertIn("failure_status=31", result.stderr)
        self.assertNotIn("ok: local Hermes voice stack verifier passed", result.stdout)

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
                    "--check-whatsapp-cloud-health",
                    "--check-whatsapp-cloud-webhook",
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
                "--check-whatsapp-cloud-health",
                "--check-whatsapp-cloud-webhook",
                "--require-complete",
                "--json",
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
                    "--skip-hermes-install-dry-run",
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
                              "--check-whatsapp-cloud-api",
                              "--check-whatsapp-cloud-health",
                              "--check-whatsapp-cloud-webhook"
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
                              "--check-whatsapp-cloud-api",
                              "--check-whatsapp-cloud-health",
                              "--check-whatsapp-cloud-webhook"
                            ],
                            "complete_verification_command": [
                              "scripts/verify_whatsapp_alpha_readiness.py",
                              "--hermes-home",
                              "/home/ubuntu/.hermes",
                              "--profile",
                              "attended-cache-receive",
                              "--require-whatsapp-cloud",
                              "--require-whatsapp-calling",
                              "--check-whatsapp-cloud-api",
                              "--check-whatsapp-cloud-health",
                              "--check-whatsapp-cloud-webhook",
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
            "/home/ubuntu/.hermes --require-whatsapp-cloud "
            "--check-whatsapp-cloud-api --check-whatsapp-cloud-health "
            "--check-whatsapp-cloud-webhook",
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
            "--require-whatsapp-calling --check-whatsapp-cloud-api "
            "--check-whatsapp-cloud-health --check-whatsapp-cloud-webhook",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_calling_complete_command="
            "scripts/verify_whatsapp_alpha_readiness.py --hermes-home "
            "/home/ubuntu/.hermes --profile attended-cache-receive "
            "--require-whatsapp-cloud --require-whatsapp-calling "
            "--check-whatsapp-cloud-api --check-whatsapp-cloud-health "
            "--check-whatsapp-cloud-webhook "
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

    def test_whatsapp_alpha_json_summary_prints_effective_actions_from_verified_watch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            watch = tmp_path / "start_watch.py"
            alpha = tmp_path / "verify_alpha.py"
            voice = tmp_path / "voice"
            json_output = tmp_path / "alpha.json"

            write_helper(whatsapp, "whatsapp", log_path)
            write_executable(
                watch,
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf 'watch' >> {str(log_path)!r}
                    printf '\\0' >> {str(log_path)!r}
                    printf '%s\\0' "$@" >> {str(log_path)!r}
                    printf '\\n' >> {str(log_path)!r}
                    if [[ "$*" == *"--json"* ]]; then
                      cat <<'JSON'
                    {{
                      "watches": [
                        {{
                          "unit": "watch-done",
                          "watch_status": "verified",
                          "alpha": {{
                            "attended_fresh_receive_verified": true,
                            "fresh_count": 1
                          }},
                          "audio_cache": {{
                            "latest_file": "aud_new.ogg",
                            "fresh_since_created": true
                          }}
                        }}
                      ]
                    }}
                    JSON
                    else
                      echo "ok: WhatsApp attended cache watch list"
                      echo "watch[1]=watch-done status=verified latest_audio=aud_new.ogg latest_audio_fresh=True"
                    fi
                    """
                ),
            )
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
                        "external_meta_setup_required": true,
                        "operator_action_required": true,
                        "next_actions": [
                          {{"id": "run_attended_fresh_receive"}},
                          {{"id": "configure_whatsapp_cloud_calling"}}
                        ]
                      }},
                      "pending_gates": {{
                        "attended_fresh_receive": {{
                          "status": "pending_attended",
                          "cached_receive_verified": true,
                          "command": [
                            "scripts/verify_whatsapp_alpha_readiness.py",
                            "--profile",
                            "attended-cache-receive"
                          ]
                        }},
                        "whatsapp_cloud": {{"status": "external_setup_required"}},
                        "whatsapp_cloud_calling": {{"status": "external_setup_required"}}
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
                    "--skip-daemon",
                    "--whatsapp-attended-watch-output-dir",
                    str(tmp_path / "watch-artifacts"),
                    "--whatsapp-attended-watch-unit-prefix",
                    "watch",
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
                    "WHATSAPP_ATTENDED_WATCH_SCRIPT": str(watch),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )
            entries = command_log_entries(log_path)

        self.assertEqual([entry[0] for entry in entries], ["whatsapp", "watch", "watch", "alpha"])
        self.assertIn(
            "whatsapp_alpha_json_next_actions="
            "run_attended_fresh_receive,configure_whatsapp_cloud_calling",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_attended_watch_evidence=verified "
            "unit=watch-done fresh_count=1 latest_audio=aud_new.ogg "
            "latest_audio_fresh=True",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_effective_next_actions="
            "configure_whatsapp_cloud_calling",
            result.stdout,
        )

    def test_whatsapp_alpha_json_output_classifies_external_meta_only_alpha(self):
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
                          "cached_receive_verified": true
                        }},
                        "whatsapp_cloud": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": ["WHATSAPP_CLOUD_PHONE_NUMBER_ID"],
                            "invalid": []
                          }}
                        }},
                        "whatsapp_cloud_calling": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": ["WHATSAPP_CLOUD_PHONE_NUMBER_ID"],
                            "invalid": []
                          }}
                        }}
                      }},
                      "failures": [
                        {{
                          "name": "whatsapp_cloud_probe",
                          "category": "external_meta_setup",
                          "failures": ["WhatsApp Cloud health check failed"]
                        }}
                      ]
                    }}
                    JSON
                    exit 1
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
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "whatsapp_alpha_json_next_actions=configure_whatsapp_cloud_calling",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha=cached-receive:external_meta_setup_pending",
            result.stdout,
        )
        self.assertIn(
            "error: WhatsApp alpha readiness profile external Meta setup pending "
            "with exit 1",
            result.stderr,
        )
        self.assertNotIn(
            "error: WhatsApp alpha readiness profile failed with exit 1",
            result.stderr,
        )

    def test_whatsapp_alpha_profile_classifies_external_meta_without_saved_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            alpha = tmp_path / "verify_alpha.py"
            voice = tmp_path / "voice"

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
                          "cached_receive_verified": true
                        }},
                        "whatsapp_cloud": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": ["WHATSAPP_CLOUD_PHONE_NUMBER_ID"],
                            "invalid": []
                          }}
                        }},
                        "whatsapp_cloud_calling": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": ["WHATSAPP_CLOUD_PHONE_NUMBER_ID"],
                            "invalid": []
                          }}
                        }}
                      }},
                      "failures": [
                        {{
                          "name": "whatsapp_cloud_probe",
                          "category": "external_meta_setup",
                          "failures": ["WhatsApp Cloud health check failed"]
                        }}
                      ]
                    }}
                    JSON
                    exit 1
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
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )

            entries = command_log_entries(log_path)

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "whatsapp_alpha_json=<temporary; pass --whatsapp-alpha-json-output to retain>",
            result.stdout,
        )
        self.assertIn("--json", entries[1])
        self.assertIn(
            "whatsapp_alpha=cached-receive:external_meta_setup_pending",
            result.stdout,
        )
        self.assertNotIn("whatsapp_alpha_json=/tmp/", result.stdout)

    def test_whatsapp_alpha_json_output_summarizes_nonzero_alpha(self):
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
                      "profile": "attended-cache-receive",
                      "readiness_summary": {{
                        "status": "local_ready_pending_gates",
                        "complete": false,
                        "local_checks_passed": true,
                        "attended_fresh_receive_verified": false,
                        "external_meta_setup_required": true,
                        "operator_action_required": true,
                        "next_actions": [
                          {{"id": "run_attended_fresh_receive"}},
                          {{"id": "configure_whatsapp_cloud_calling"}}
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
                          ]
                        }},
                        "whatsapp_cloud": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": ["WHATSAPP_CLOUD_PHONE_NUMBER_ID"],
                            "invalid": []
                          }}
                        }},
                        "whatsapp_cloud_calling": {{
                          "status": "external_setup_required",
                          "setup_handoff": {{
                            "missing": ["WHATSAPP_CLOUD_PHONE_NUMBER_ID"],
                            "invalid": [],
                            "complete_verification_command": [
                              "scripts/verify_whatsapp_alpha_readiness.py",
                              "--hermes-home",
                              "/home/ubuntu/.hermes",
                              "--profile",
                              "attended-cache-receive",
                              "--require-whatsapp-cloud",
                              "--require-whatsapp-calling",
                              "--check-whatsapp-cloud-api",
                              "--check-whatsapp-cloud-health",
                              "--check-whatsapp-cloud-webhook",
                              "--require-complete"
                            ]
                          }}
                        }}
                      }}
                    }}
                    JSON
                    exit 3
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
                    "attended-cache-receive",
                    "--require-whatsapp-alpha-complete",
                    "--whatsapp-alpha-json-output",
                    str(json_output),
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "WHATSAPP_ALPHA_READINESS_SCRIPT": str(alpha),
                },
            )

        self.assertEqual(result.returncode, 3)
        self.assertIn(f"whatsapp_alpha_json={json_output}", result.stdout)
        self.assertIn("whatsapp_alpha_json_profile=attended-cache-receive", result.stdout)
        self.assertIn(
            "whatsapp_alpha_json_readiness=local_ready_pending_gates complete=False",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_next_actions=run_attended_fresh_receive,"
            "configure_whatsapp_cloud_calling",
            result.stdout,
        )
        self.assertIn(
            "whatsapp_alpha_json_attended_command="
            "scripts/verify_whatsapp_alpha_readiness.py --hermes-home "
            "/home/ubuntu/.hermes --profile attended-cache-receive "
            "--wait-audio-cache-seconds 60.0",
            result.stdout,
        )
        self.assertIn("whatsapp_alpha=attended-cache-receive:failed", result.stdout)
        self.assertNotIn("ok: local Hermes voice stack verifier passed", result.stdout)
        self.assertIn(
            "error: WhatsApp alpha readiness profile failed with exit 3",
            result.stderr,
        )
        self.assertIn(
            "error: local Hermes voice stack verifier failed",
            result.stderr,
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
                    "--skip-hermes-install-dry-run",
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
