#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "install_webrtc_sidecar_service.sh"


class WebrtcSidecarServiceInstallTests(unittest.TestCase):
    def test_print_unit_renders_pinned_local_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            venv = tmp_path / "voice-webrtc-venv"
            rx_pcm = tmp_path / "state" / "voice" / "inbound.s16le"

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--print-unit",
                    "--repo-root",
                    str(REPO_ROOT),
                    "--venv",
                    str(venv),
                    "--voice-bin",
                    "/opt/voice/bin/voice",
                    "--rx-pcm",
                    str(rx_pcm),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9999",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HOME": str(tmp_path),
                    "XDG_CONFIG_HOME": str(tmp_path / "config"),
                    "XDG_DATA_HOME": str(tmp_path / "data"),
                    "XDG_STATE_HOME": str(tmp_path / "state"),
                },
            )

        unit = result.stdout
        self.assertIn("Description=Voice WebRTC sidecar", unit)
        self.assertIn("After=voiced.service", unit)
        self.assertIn("Wants=voiced.service", unit)
        self.assertIn('Environment="VOICE_BIN=/opt/voice/bin/voice"', unit)
        self.assertIn(f"WorkingDirectory={REPO_ROOT}", unit)
        self.assertIn(
            f'ExecStart="{venv}/bin/python" '
            f'"{REPO_ROOT}/examples/webrtc-sidecar/sidecar.py" '
            f'--host "127.0.0.1" --port 9999 --rx-pcm "{rx_pcm}" '
            '--log-level "INFO"',
            unit,
        )
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("WantedBy=default.target", unit)

    def test_print_unit_rejects_non_loopback_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            result = subprocess.run(
                [
                    str(SCRIPT_PATH),
                    "--print-unit",
                    "--repo-root",
                    str(REPO_ROOT),
                    "--voice-bin",
                    "/opt/voice/bin/voice",
                    "--host",
                    "0.0.0.0",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HOME": str(tmp_path),
                    "XDG_CONFIG_HOME": str(tmp_path / "config"),
                    "XDG_DATA_HOME": str(tmp_path / "data"),
                    "XDG_STATE_HOME": str(tmp_path / "state"),
                },
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing non-loopback --host '0.0.0.0'", result.stderr)


if __name__ == "__main__":
    unittest.main()
