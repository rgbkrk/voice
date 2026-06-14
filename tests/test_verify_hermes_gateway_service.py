#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_hermes_gateway_service.py"


def write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def write_fake_systemctl(
    path: Path,
    *,
    hermes_home: Path,
    voice_bin: Path,
    pythonpath: Path,
    stream_command: str | None = None,
    active_state: str = "active",
) -> None:
    command = stream_command or (
        f"{voice_bin} stream --quiet --sample-rate {{sample_rate}} "
        "--frame-ms {frame_ms} --raw-output - --input-file {input_path} "
        "--voice af_heart --speed 1.0"
    )
    body = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cat <<'EOF'
        ActiveState={active_state}
        SubState=running
        MainPID=12345
        ExecStart={{ path={hermes_home}/hermes-agent/venv/bin/python ; argv[]={hermes_home}/hermes-agent/venv/bin/python -m hermes_cli.main gateway run ; ignore_errors=no ; }}
        Environment=HERMES_HOME={hermes_home} PYTHONPATH={pythonpath} WHATSAPP_CLOUD_CALLING_SIDECAR_URL=http://127.0.0.1:8787 "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND={command}" WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT=180
        WorkingDirectory={hermes_home}
        FragmentPath={hermes_home}/systemd/hermes-gateway.service
        DropInPaths={hermes_home}/systemd/hermes-gateway.service.d/voice-stack.conf
        EOF
        """
    )
    write_executable(path, body)


class HermesGatewayServiceVerifierTests(unittest.TestCase):
    def test_verifier_accepts_voice_native_gateway_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hermes_home = tmp_path / "hermes"
            pythonpath = tmp_path / "hermes-agent-voice-stack"
            bin_dir = tmp_path / "bin"
            hermes_home.mkdir()
            pythonpath.mkdir()
            bin_dir.mkdir()
            voice = bin_dir / "voice"
            systemctl = bin_dir / "systemctl"
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            write_fake_systemctl(
                systemctl,
                hermes_home=hermes_home,
                voice_bin=voice,
                pythonpath=pythonpath,
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-home",
                    str(hermes_home),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            )

        payload = json.loads(result.stdout)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["failures"], [])
        self.assertEqual(
            payload["checks"]["gateway_service"]["sidecar_url"],
            "http://127.0.0.1:8787",
        )

    def test_verifier_rejects_gateway_without_raw_stream_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hermes_home = tmp_path / "hermes"
            pythonpath = tmp_path / "hermes-agent-voice-stack"
            bin_dir = tmp_path / "bin"
            hermes_home.mkdir()
            pythonpath.mkdir()
            bin_dir.mkdir()
            voice = bin_dir / "voice"
            systemctl = bin_dir / "systemctl"
            write_executable(voice, "#!/usr/bin/env bash\nexit 0\n")
            write_fake_systemctl(
                systemctl,
                hermes_home=hermes_home,
                voice_bin=voice,
                pythonpath=pythonpath,
                stream_command=f"{voice} stream --quiet --input-file {{input_path}}",
            )

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--voice-bin",
                    str(voice),
                    "--hermes-home",
                    str(hermes_home),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stream command must pass --raw-output -", result.stderr)


if __name__ == "__main__":
    unittest.main()
