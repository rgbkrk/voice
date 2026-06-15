#!/usr/bin/env python3

from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
    def test_dry_run_prefers_installed_voice_on_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            fake_voice = bin_dir / "voice"
            write_executable(
                fake_voice,
                "#!/usr/bin/env bash\n"
                "echo fake voice\n",
            )
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            }
            env.pop("VOICE_BIN", None)

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--json",
                    "--timestamp",
                    "20260614T225118Z",
                    "--output-dir",
                    str(tmp_path),
                    "--hermes-home",
                    str(tmp_path / "hermes"),
                    "--hermes-config",
                    str(tmp_path / "hermes" / "config.yaml"),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["manifest"]["voice_bin"], str(fake_voice))
        self.assertIn(str(fake_voice), payload["alpha_command"])

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
        self.assertTrue(payload["manifest_path"].endswith(".manifest.json"))
        self.assertFalse(Path(payload["manifest_path"]).exists())
        manifest = payload["manifest"]
        self.assertEqual(
            manifest["schema"],
            "voice.whatsapp_attended_cache_watch_manifest",
        )
        self.assertEqual(manifest["profile"], "attended-cache-receive")
        self.assertFalse(manifest["drains_bridge_messages"])
        self.assertEqual(manifest["wait_seconds"], 120.0)
        self.assertEqual(manifest["expected_agent_number"], "13236478455")
        self.assertEqual(manifest["expected_agent_name"], "Quill")
        prompt = manifest["attended_prompt"]
        self.assertTrue(prompt["sends_prompt_voice_note"])
        self.assertEqual(
            prompt["prompt_text"],
            "Please reply with a fresh WhatsApp voice note so I can verify the voice runtime.",
        )
        self.assertEqual(prompt["send_format"], "audio/ogg; codecs=opus")
        self.assertEqual(prompt["send_transport"], "local_whatsapp_bridge_ptt")
        self.assertEqual(prompt["receive_watch"], "non_draining_audio_cache")
        self.assertEqual(
            prompt["audio_cache_dir"],
            str(tmp_path / "hermes" / "audio_cache"),
        )
        self.assertEqual(manifest["artifacts"]["json"], payload["json_path"])
        self.assertEqual(manifest["artifacts"]["log"], payload["log_path"])
        self.assertEqual(manifest["artifacts"]["manifest"], payload["manifest_path"])
        alpha = payload["alpha_command"]
        self.assertIn("--profile", alpha)
        self.assertIn("attended-cache-receive", alpha)
        self.assertIn("--attended-prompt-text", alpha)
        self.assertIn(prompt["prompt_text"], alpha)
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
            manifest_path = (
                tmp_path
                / "voice-whatsapp-attended-cache-watch-20260614T225118Z.manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIn("ok: WhatsApp attended cache watch started", result.stdout)
        self.assertIn("wait_seconds=30.0", result.stdout)
        self.assertIn("manifest=", result.stdout)
        self.assertEqual(manifest["unit"], "voice-whatsapp-attended-cache-watch-20260614T225118Z")
        self.assertEqual(manifest["wait_seconds"], 30.0)
        self.assertFalse(manifest["drains_bridge_messages"])
        self.assertEqual(manifest["profile"], "attended-cache-receive")
        self.assertTrue(manifest["attended_prompt"]["sends_prompt_voice_note"])
        self.assertEqual(
            manifest["attended_prompt"]["audio_cache_dir"],
            str(tmp_path / "hermes" / "audio_cache"),
        )
        self.assertIn("alpha", manifest["commands"])
        self.assertIn("--wait-audio-cache-seconds", manifest["commands"]["alpha"])
        self.assertIn("--attended-prompt-text", manifest["commands"]["alpha"])
        self.assertEqual(
            manifest["artifacts"]["json"],
            str(tmp_path / "voice-whatsapp-attended-cache-watch-20260614T225118Z.json"),
        )
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

    def test_status_reports_active_empty_artifact_as_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_systemctl = tmp_path / "systemctl"
            write_executable(
                fake_systemctl,
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    cat <<'EOF'
                    ActiveState=active
                    SubState=running
                    MainPID=12345
                    EOF
                    """
                ),
            )
            (tmp_path / "watch.json").write_text("", encoding="utf-8")
            (tmp_path / "watch.log").write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--status",
                    "watch.service",
                    "--output-dir",
                    str(tmp_path),
                    "--systemctl-bin",
                    str(fake_systemctl),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["watch_status"], "waiting_for_fresh_audio")
        self.assertEqual(payload["systemd"]["ActiveState"], "active")
        self.assertEqual(payload["systemd"]["MainPID"], "12345")
        self.assertTrue(payload["json"]["exists"])
        self.assertEqual(payload["json"]["size_bytes"], 0)
        self.assertFalse(payload["manifest"]["exists"])
        self.assertEqual(payload["manifest"]["size_bytes"], 0)
        self.assertEqual(payload["manifest_summary"], {})
        self.assertEqual(payload["service"], "watch.service")

    def test_status_summarizes_verified_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_systemctl = tmp_path / "systemctl"
            write_executable(
                fake_systemctl,
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    cat <<'EOF'
                    ActiveState=inactive
                    SubState=dead
                    MainPID=0
                    EOF
                    """
                ),
            )
            (tmp_path / "watch.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "profile": "attended-cache-receive",
                        "readiness_summary": {
                            "status": "local_ready_pending_gates",
                            "complete": False,
                            "attended_fresh_receive_verified": True,
                            "external_meta_setup_required": True,
                        },
                        "pending_gates": {
                            "attended_fresh_receive": {
                                "status": "verified",
                                "cached_receive_verified": True,
                                "evidence": {
                                    "kind": "audio_cache",
                                    "fresh_count": 1,
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "watch.log").write_text("done\n", encoding="utf-8")
            audio_cache = tmp_path / "audio_cache"
            audio_cache.mkdir()
            stale_audio = audio_cache / "aud_stale.ogg"
            stale_audio.write_bytes(b"stale")
            fresh_audio = audio_cache / "aud_fresh.ogg"
            fresh_audio.write_bytes(b"fresh-audio")
            stale_time = datetime(2026, 6, 14, 22, 50, tzinfo=timezone.utc).timestamp()
            fresh_time = datetime(2026, 6, 14, 22, 52, tzinfo=timezone.utc).timestamp()
            os.utime(stale_audio, (stale_time, stale_time))
            os.utime(fresh_audio, (fresh_time, fresh_time))
            (tmp_path / "watch.manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "voice.whatsapp_attended_cache_watch_manifest",
                        "version": 1,
                        "profile": "attended-cache-receive",
                        "created_at_utc": "2026-06-14T22:51:18Z",
                        "wait_seconds": 120.0,
                        "drains_bridge_messages": False,
                        "attended_prompt": {
                            "sends_prompt_voice_note": True,
                            "prompt_text": "reply with a voice note",
                            "audio_cache_dir": str(audio_cache),
                        },
                        "expected_agent_number": "13236478455",
                        "expected_agent_name": "Quill",
                        "artifacts": {
                            "json": str(tmp_path / "watch.json"),
                            "log": str(tmp_path / "watch.log"),
                            "manifest": str(tmp_path / "watch.manifest.json"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--status",
                    "watch",
                    "--output-dir",
                    str(tmp_path),
                    "--systemctl-bin",
                    str(fake_systemctl),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["watch_status"], "verified")
        self.assertTrue(payload["json"]["parsed"])
        self.assertEqual(payload["alpha"]["profile"], "attended-cache-receive")
        self.assertEqual(payload["alpha"]["attended_status"], "verified")
        self.assertTrue(payload["alpha"]["attended_fresh_receive_verified"])
        self.assertEqual(payload["alpha"]["fresh_count"], 1)
        self.assertTrue(payload["manifest"]["exists"])
        self.assertTrue(payload["manifest"]["parsed"])
        self.assertEqual(
            payload["manifest_summary"]["profile"],
            "attended-cache-receive",
        )
        self.assertEqual(payload["manifest_summary"]["wait_seconds"], 120.0)
        self.assertFalse(payload["manifest_summary"]["drains_bridge_messages"])
        self.assertEqual(
            payload["manifest_summary"]["attended_prompt"]["prompt_text"],
            "reply with a voice note",
        )
        self.assertEqual(payload["timing"]["created_at_utc"], "2026-06-14T22:51:18Z")
        self.assertEqual(payload["timing"]["deadline_utc"], "2026-06-14T22:53:18Z")
        self.assertEqual(payload["timing"]["wait_seconds"], 120.0)
        self.assertTrue(payload["timing"]["expired"])
        self.assertEqual(payload["audio_cache"]["path"], str(audio_cache))
        self.assertTrue(payload["audio_cache"]["exists"])
        self.assertEqual(payload["audio_cache"]["candidate_count"], 2)
        self.assertEqual(payload["audio_cache"]["latest_file"], "aud_fresh.ogg")
        self.assertEqual(payload["audio_cache"]["latest_path"], str(fresh_audio))
        self.assertEqual(
            payload["audio_cache"]["latest_mtime_utc"],
            "2026-06-14T22:52:00Z",
        )
        self.assertEqual(
            payload["audio_cache"]["latest_size_bytes"],
            len(b"fresh-audio"),
        )
        self.assertTrue(payload["audio_cache"]["fresh_since_created"])

    def test_list_discovers_active_units_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_systemctl = tmp_path / "systemctl"
            write_executable(
                fake_systemctl,
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    args="$*"
                    if [[ "$args" == *"list-units"* ]]; then
                      echo "watch-active.service loaded active running active watch"
                      exit 0
                    fi
                    if [[ "$args" == *"show watch-active.service"* ]]; then
                      cat <<'EOF'
                    ActiveState=active
                    SubState=running
                    MainPID=12345
                    EOF
                      exit 0
                    fi
                    if [[ "$args" == *"show watch-done.service"* ]]; then
                      cat <<'EOF'
                    ActiveState=inactive
                    SubState=dead
                    MainPID=0
                    EOF
                      exit 0
                    fi
                    if [[ "$args" == *"show watch-manifest.service"* ]]; then
                      cat <<'EOF'
                    ActiveState=inactive
                    SubState=dead
                    MainPID=0
                    EOF
                      exit 0
                    fi
                    echo "unknown args: $args" >&2
                    exit 1
                    """
                ),
            )
            (tmp_path / "watch-active.json").write_text("", encoding="utf-8")
            (tmp_path / "watch-active.log").write_text("", encoding="utf-8")
            (tmp_path / "watch-done.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "profile": "attended-cache-receive",
                        "readiness_summary": {
                            "attended_fresh_receive_verified": True,
                        },
                        "pending_gates": {
                            "attended_fresh_receive": {
                                "status": "verified",
                                "evidence": {
                                    "kind": "audio_cache",
                                    "fresh_count": 1,
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "watch-manifest.manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "voice.whatsapp_attended_cache_watch_manifest",
                        "version": 1,
                        "profile": "attended-cache-receive",
                        "created_at_utc": "2026-06-14T22:51:18Z",
                        "wait_seconds": 60.0,
                        "drains_bridge_messages": False,
                        "attended_prompt": {
                            "sends_prompt_voice_note": True,
                            "prompt_text": "reply with a voice note",
                            "audio_cache_dir": str(tmp_path / "audio_cache"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--list",
                    "--unit-prefix",
                    "watch",
                    "--output-dir",
                    str(tmp_path),
                    "--systemctl-bin",
                    str(fake_systemctl),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["count"], 3)
        by_unit = {watch["unit"]: watch for watch in payload["watches"]}
        self.assertEqual(
            by_unit["watch-active"]["watch_status"],
            "waiting_for_fresh_audio",
        )
        self.assertEqual(by_unit["watch-done"]["watch_status"], "verified")
        self.assertEqual(by_unit["watch-done"]["alpha"]["fresh_count"], 1)
        self.assertEqual(by_unit["watch-manifest"]["watch_status"], "no_artifact")
        self.assertTrue(by_unit["watch-manifest"]["manifest"]["parsed"])
        self.assertEqual(
            by_unit["watch-manifest"]["manifest_summary"]["wait_seconds"],
            60.0,
        )
        self.assertEqual(
            by_unit["watch-manifest"]["timing"]["deadline_utc"],
            "2026-06-14T22:52:18Z",
        )
        self.assertTrue(
            by_unit["watch-manifest"]["manifest_summary"]["attended_prompt"][
                "sends_prompt_voice_note"
            ]
        )

    def test_stop_requests_systemd_stop_and_reports_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "systemctl.log"
            fake_systemctl = tmp_path / "systemctl"
            write_executable(
                fake_systemctl,
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\0' "$@" >> {str(log_path)!r}
                    printf '\\n' >> {str(log_path)!r}
                    args="$*"
                    if [[ "$args" == *" stop watch.service" ]]; then
                      exit 0
                    fi
                    if [[ "$args" == *"show watch.service"* ]]; then
                      cat <<'EOF'
                    ActiveState=inactive
                    SubState=dead
                    MainPID=0
                    EOF
                      exit 0
                    fi
                    echo "unknown args: $args" >&2
                    exit 1
                    """
                ),
            )
            (tmp_path / "watch.json").write_text("", encoding="utf-8")
            (tmp_path / "watch.log").write_text("", encoding="utf-8")
            (tmp_path / "watch.manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "voice.whatsapp_attended_cache_watch_manifest",
                        "version": 1,
                        "profile": "attended-cache-receive",
                        "wait_seconds": 30.0,
                        "drains_bridge_messages": False,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--stop",
                    "watch",
                    "--output-dir",
                    str(tmp_path),
                    "--systemctl-bin",
                    str(fake_systemctl),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            entries = command_log_entries(log_path)

        payload = json.loads(result.stdout)
        self.assertEqual(payload["stop_returncode"], 0)
        self.assertEqual(payload["watch_status"], "empty_artifact")
        self.assertTrue(payload["json"]["exists"])
        self.assertTrue(payload["manifest"]["exists"])
        self.assertEqual(payload["manifest_summary"]["wait_seconds"], 30.0)
        self.assertEqual(entries[0], ["--user", "stop", "watch.service"])
        self.assertIn("show", entries[1])

    def test_stop_failure_returns_systemctl_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_systemctl = tmp_path / "systemctl"
            write_executable(
                fake_systemctl,
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    args="$*"
                    if [[ "$args" == *" stop watch.service" ]]; then
                      echo "unit missing" >&2
                      exit 4
                    fi
                    if [[ "$args" == *"show watch.service"* ]]; then
                      cat <<'EOF'
                    ActiveState=inactive
                    SubState=dead
                    MainPID=0
                    EOF
                      exit 0
                    fi
                    exit 1
                    """
                ),
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--stop",
                    "watch",
                    "--output-dir",
                    str(tmp_path),
                    "--systemctl-bin",
                    str(fake_systemctl),
                    "--json",
                ],
                capture_output=True,
                text=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 4)
        self.assertEqual(payload["stop_returncode"], 4)
        self.assertIn("unit missing", payload["stop_stderr"])
        self.assertEqual(payload["watch_status"], "no_artifact")

    def test_status_list_and_stop_cannot_be_combined(self):
        result = subprocess.run(
            [
                str(SCRIPT_PATH),
                "--status",
                "watch",
                "--list",
                "--stop",
                "watch",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--status, --list, and --stop are mutually exclusive",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
