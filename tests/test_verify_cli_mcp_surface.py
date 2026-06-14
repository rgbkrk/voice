#!/usr/bin/env python3

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_cli_mcp_surface.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("verify_cli_mcp_surface", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CliMcpSurfaceVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script_module()

    def make_fake_voice(self, tmp_path: Path) -> Path:
        fake = tmp_path / "voice"
        fake.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import sys

                args = sys.argv[1:]
                if args == ["stream-contract"]:
                    print(json.dumps({"contract": "voice.webrtc_sidecar"}))
                    raise SystemExit(0)
                if args == ["daemon", "status", "--json"]:
                    if os.environ.get("VOICE_FAKE_DAEMON") == "1":
                        print(json.dumps({"running": True}))
                        raise SystemExit(0)
                    print("daemon not running", file=sys.stderr)
                    raise SystemExit(1)
                if args and args[0] == "mcp":
                    quiet = "-q" in args
                    if not quiet and os.environ.get("VOICE_FAKE_DAEMON") == "1":
                        print("voice mcp: connected to voice daemon", file=sys.stderr)
                    print('{"result":{"serverInfo":{"name":"voice"}},"id":1}')
                    print('{"result":{"tools":[]},"id":2}')
                    raise SystemExit(0)
                print(f"unexpected args: {args}", file=sys.stderr)
                raise SystemExit(2)
                """
            ),
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return fake

    def verify_with_fake_voice(
        self,
        fake_voice: Path,
        *,
        require_daemon: bool = False,
        skip_daemon: bool = False,
    ):
        return self.script.verify(
            argparse.Namespace(
                voice_bin=fake_voice,
                require_daemon=require_daemon,
                skip_daemon=skip_daemon,
                timeout=5.0,
            )
        )

    def test_no_daemon_surfaces_pass_and_daemon_smoke_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_voice = self.make_fake_voice(Path(tmp))

            result = self.verify_with_fake_voice(fake_voice)

        self.assertTrue(result["success"])
        checks = {check["name"]: check for check in result["checks"]}
        self.assertTrue(checks["stream_contract_no_daemon"]["ok"])
        self.assertTrue(checks["mcp_no_daemon_initializes"]["ok"])
        self.assertFalse(checks["daemon_detected"]["detected"])
        self.assertTrue(checks["mcp_with_daemon_detects_daemon"]["skipped"])

    def test_require_daemon_fails_when_daemon_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_voice = self.make_fake_voice(Path(tmp))

            result = self.verify_with_fake_voice(fake_voice, require_daemon=True)

        self.assertFalse(result["success"])
        checks = {check["name"]: check for check in result["checks"]}
        self.assertFalse(checks["daemon_detected"]["ok"])
        self.assertFalse(checks["daemon_detected"]["detected"])
        self.assertFalse(checks["mcp_with_daemon_detects_daemon"]["ok"])

    def test_daemon_detected_path_requires_mcp_connection_log(self):
        old_value = os.environ.get("VOICE_FAKE_DAEMON")
        os.environ["VOICE_FAKE_DAEMON"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fake_voice = self.make_fake_voice(Path(tmp))

                result = self.verify_with_fake_voice(fake_voice)
        finally:
            if old_value is None:
                os.environ.pop("VOICE_FAKE_DAEMON", None)
            else:
                os.environ["VOICE_FAKE_DAEMON"] = old_value

        self.assertTrue(result["success"])
        checks = {check["name"]: check for check in result["checks"]}
        self.assertTrue(checks["daemon_detected"]["detected"])
        self.assertTrue(checks["mcp_with_daemon_detects_daemon"]["ok"])


if __name__ == "__main__":
    unittest.main()
