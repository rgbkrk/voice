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
            whatsapp = tmp_path / "verify_whatsapp.sh"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_helper(hermes, "hermes", log_path)
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
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("ok: local Hermes voice stack verifier passed", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["hermes", "whatsapp", "sidecar"])
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
            entries[2],
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
            whatsapp = tmp_path / "verify_whatsapp.sh"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"
            config = tmp_path / "config.yaml"

            write_helper(hermes, "hermes", log_path)
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
                    "--skip-sidecar",
                    "--skip-daemon",
                    "--skip-stt-smoke",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("sidecar_service=skipped", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["hermes", "whatsapp"])
        self.assertIn("--skip-tts-smoke", entries[0])
        self.assertIn("--skip-daemon", entries[1])
        self.assertNotIn("--run-stt-smoke", entries[1])

    def test_skip_hermes_config_does_not_require_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "commands.log"
            hermes = tmp_path / "verify_hermes.py"
            whatsapp = tmp_path / "verify_whatsapp.sh"
            sidecar = tmp_path / "verify_sidecar.py"
            voice = tmp_path / "voice"

            write_helper(hermes, "hermes", log_path)
            write_helper(whatsapp, "whatsapp", log_path)
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
                    "--skip-sidecar",
                    "--skip-daemon",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HERMES_CONFIG_VERIFY_SCRIPT": str(hermes),
                    "WHATSAPP_CONTRACT_VERIFY_SCRIPT": str(whatsapp),
                    "SIDECAR_SERVICE_VERIFY_SCRIPT": str(sidecar),
                },
            )

            entries = command_log_entries(log_path)

        self.assertIn("hermes_config=skipped", result.stdout)
        self.assertEqual([entry[0] for entry in entries], ["whatsapp"])


if __name__ == "__main__":
    unittest.main()
