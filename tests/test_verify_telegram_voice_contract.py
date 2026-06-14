#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_telegram_voice_contract.sh"


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


def command_log_entries(log_path: Path) -> list[list[str]]:
    entries: list[list[str]] = []
    for line in log_path.read_bytes().splitlines():
        if not line:
            continue
        entries.append(line.decode("utf-8").split("\0")[:-1])
    return entries


class TelegramVoiceContractVerifierTests(unittest.TestCase):
    def test_runs_voice_contract_and_hermes_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            voice_contract = tmp_path / "verify_voice_contract.sh"
            hermes_config_verifier = tmp_path / "verify_hermes.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"
            env_file = tmp_path / ".env"

            write_helper(voice_contract, "voice_contract", log_path)
            write_helper(hermes_config_verifier, "hermes", log_path)
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            config.write_text("tts: {}\n", encoding="utf-8")
            env_file.write_text(
                "\n".join(
                    [
                        "# Telegram setup",
                        "TELEGRAM_BOT_TOKEN=123:abc",
                        "TELEGRAM_ALLOWED_USERS=42,43",
                        "TELEGRAM_HOME_CHANNEL=-100123",
                        "TELEGRAM_WEBHOOK_URL=https://example.invalid/telegram",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--text",
                    "Telegram smoke.",
                    "--hermes-config",
                    str(config),
                    "--hermes-env",
                    str(env_file),
                    "--skip-hermes-tts-smoke",
                    "--skip-daemon",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "VOICE_CONTRACT_VERIFY_SCRIPT": str(voice_contract),
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes_config_verifier),
                },
            )

            entries = command_log_entries(log_path) if log_path.exists() else []

        self.assertIn("ok: voice Telegram contract verifier passed", result.stdout)
        self.assertIn("voice_contract=checked", result.stdout)
        self.assertIn("hermes_voice_config=checked_without_tts_smoke", result.stdout)
        self.assertIn(f"telegram_env={env_file}", result.stdout)
        self.assertIn("telegram_env_status=found", result.stdout)
        self.assertIn("telegram_credentials=configured", result.stdout)
        self.assertIn("telegram_allowed_users=configured", result.stdout)
        self.assertIn("telegram_home_channel=configured", result.stdout)
        self.assertIn("telegram_webhook=configured", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["voice_contract", "hermes"])
        self.assertEqual(
            entries[0],
            [
                "voice_contract",
                "--text",
                "Telegram smoke.",
                "--voice-bin",
                str(voice),
                "--skip-daemon",
            ],
        )
        self.assertEqual(
            entries[1],
            [
                "hermes",
                "--config",
                str(config),
                "--voice-bin",
                str(voice),
                "--skip-tts-smoke",
            ],
        )

    def test_skip_hermes_config_only_runs_voice_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            voice_contract = tmp_path / "verify_voice_contract.sh"
            missing_config = tmp_path / "missing.yaml"
            missing_env = tmp_path / "missing.env"

            write_helper(voice_contract, "voice_contract", log_path)

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--hermes-config",
                    str(missing_config),
                    "--hermes-env",
                    str(missing_env),
                    "--skip-hermes-config",
                    "--require-daemon",
                    "--run-stt-smoke",
                    "--keep-output",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "VOICE_CONTRACT_VERIFY_SCRIPT": str(voice_contract),
                },
            )

            entries = command_log_entries(log_path) if log_path.exists() else []

        self.assertIn("hermes_voice_config=skipped", result.stdout)
        self.assertIn(f"telegram_env={missing_env}", result.stdout)
        self.assertIn("telegram_env_status=missing", result.stdout)
        self.assertIn("telegram_credentials=not_configured", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["voice_contract"])
        self.assertIn("--require-daemon", entries[0])
        self.assertIn("--run-stt-smoke", entries[0])
        self.assertIn("--keep-output", entries[0])

    def test_require_telegram_credentials_fails_without_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            voice_contract = tmp_path / "verify_voice_contract.sh"
            env_file = tmp_path / ".env"

            write_helper(voice_contract, "voice_contract", log_path)
            env_file.write_text(
                "# TELEGRAM_BOT_TOKEN=\nTELEGRAM_ALLOWED_USERS=42\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--hermes-env",
                    str(env_file),
                    "--skip-hermes-config",
                    "--require-telegram-credentials",
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "VOICE_CONTRACT_VERIFY_SCRIPT": str(voice_contract),
                },
            )

            entries = command_log_entries(log_path) if log_path.exists() else []

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TELEGRAM_BOT_TOKEN is missing or empty", result.stderr)
        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
