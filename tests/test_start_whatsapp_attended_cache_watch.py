#!/usr/bin/env python3

from pathlib import Path
import json
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "start_whatsapp_attended_cache_watch.py"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def command_log_entries(log_path: Path) -> list[list[str]]:
    entries: list[list[str]] = []
    for line in log_path.read_bytes().splitlines():
        if not line:
            continue
        entries.append(line.decode("utf-8").split("\0")[:-1])
    return entries


class WhatsAppAttendedCacheWatchLauncherTests(unittest.TestCase):
    def test_dry_run_reports_repeatable_unit_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--json",
                    "--timestamp",
                    "20260614T225118Z",
                    "--output-dir",
                    str(tmp_path),
                    "--voice-bin",
                    "/opt/voice/bin/voice",
                    "--hermes-home",
                    str(tmp_path / "hermes"),
                    "--hermes-config",
                    str(tmp_path / "hermes" / "config.yaml"),
                    "--wait-seconds",
                    "120",
                    "--expected-agent-number",
                    "13236478455",
                    "--expected-agent-name",
                    "Quill",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(
            payload["unit"],
            "voice-whatsapp-attended-cache-watch-20260614T225118Z",
        )
        self.assertTrue(payload["json_path"].endswith(".json"))
        self.assertTrue(payload["log_path"].endswith(".log"))
        alpha = payload["alpha_command"]
        self.assertIn("--profile", alpha)
        self.assertIn("attended-cache-receive", alpha)
        self.assertIn("--wait-audio-cache-seconds", alpha)
        self.assertIn("120.0", alpha)
        self.assertIn("--expected-agent-number", alpha)
        self.assertIn("13236478455", alpha)
        systemd = payload["systemd_command"]
        self.assertIn("--collect", systemd)
        self.assertIn(
            "--unit=voice-whatsapp-attended-cache-watch-20260614T225118Z",
            systemd,
        )
        self.assertIn("verify_whatsapp_alpha_readiness.py", systemd[-1])
        self.assertIn(payload["json_path"], systemd[-1])
        self.assertIn(payload["log_path"], systemd[-1])

    def test_start_invokes_systemd_run_with_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "systemd-run.log"
            fake_systemd_run = tmp_path / "systemd-run"
            write_executable(
                fake_systemd_run,
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf 'systemd-run' >> {str(log_path)!r}
                    printf '\\0' >> {str(log_path)!r}
                    printf '%s\\0' "$@" >> {str(log_path)!r}
                    printf '\\n' >> {str(log_path)!r}
                    echo "Running as unit: $2.service"
                    """
                ),
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--systemd-run-bin",
                    str(fake_systemd_run),
                    "--timestamp",
                    "20260614T225118Z",
                    "--output-dir",
                    str(tmp_path),
                    "--voice-bin",
                    "/opt/voice/bin/voice",
                    "--hermes-home",
                    str(tmp_path / "hermes"),
                    "--hermes-config",
                    str(tmp_path / "hermes" / "config.yaml"),
                    "--wait-seconds",
                    "30",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            entries = command_log_entries(log_path)

        self.assertIn("ok: WhatsApp attended cache watch started", result.stdout)
        self.assertIn("wait_seconds=30.0", result.stdout)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry[0], "systemd-run")
        self.assertIn("--user", entry)
        self.assertIn(
            "--unit=voice-whatsapp-attended-cache-watch-20260614T225118Z",
            entry,
        )
        self.assertIn("--collect", entry)
        self.assertEqual(entry[-2], "-lc")
        self.assertIn("--wait-audio-cache-seconds 30.0", entry[-1])
        self.assertIn(".json", entry[-1])
        self.assertIn(".log", entry[-1])

    def test_rejects_non_positive_wait_window(self):
        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--dry-run",
                "--wait-seconds",
                "0",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--wait-seconds must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
