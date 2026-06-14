#!/usr/bin/env python3

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "macos_release_compare.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("macos_release_compare", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArticulationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script_module()

    def test_missing_expected_words_accounts_for_repeated_words(self):
        transcript = "Wait what, wait what?"

        missing = self.script.missing_expected_words(
            transcript,
            ["wait", "wait", "what", "what"],
        )

        self.assertEqual(missing, [])

    def test_missing_expected_words_reports_missing_duplicate(self):
        transcript = "Wait, wait, what?"

        missing = self.script.missing_expected_words(
            transcript,
            ["wait", "wait", "what", "what"],
        )

        self.assertEqual(missing, ["what"])

    def test_expected_word_must_normalize_to_one_token(self):
        with self.assertRaisesRegex(ValueError, "normalize to one token"):
            self.script.missing_expected_words("Wait what", ["wait what"])

    def test_mcp_initialized_requires_initialize_and_tools_output(self):
        self.assertTrue(
            self.script.mcp_initialized(
                '{"result":{"serverInfo":{"name":"voice"}},"id":1}\n'
                '{"result":{"tools":[]},"id":2}\n'
            )
        )
        self.assertFalse(self.script.mcp_initialized('{"result":{"serverInfo":{}}}'))

    def test_mcp_connected_to_daemon_accepts_startup_and_reconnect_logs(self):
        self.assertTrue(
            self.script.mcp_connected_to_daemon(
                "voice mcp: connected to voice daemon\nvoice mcp server ready\n"
            )
        )
        self.assertTrue(
            self.script.mcp_connected_to_daemon(
                "voice mcp: reconnected to voice daemon\n"
            )
        )
        self.assertFalse(self.script.mcp_connected_to_daemon("voice mcp server ready\n"))

    def make_fake_voice(self, tmp_path: Path) -> Path:
        fake = tmp_path / "voice"
        fake.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                from pathlib import Path
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
                if args and args[0] == "say":
                    out = Path(args[args.index("-o") + 1])
                    out.write_bytes(b"RIFF" + (b"0" * 64))
                    raise SystemExit(0)
                if args and args[0] == "stream":
                    if os.environ.get("VOICE_FAKE_DAEMON") == "1":
                        out = Path(args[args.index("-o") + 1])
                        out.write_bytes(b"pcm")
                        raise SystemExit(0)
                    print("daemon not running", file=sys.stderr)
                    raise SystemExit(1)
                print(f"unexpected args: {args}", file=sys.stderr)
                raise SystemExit(2)
                """
            ),
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return fake

    def smoke_checks_with_fake_voice(self, fake_voice: Path, work_dir: Path):
        return self.script.smoke_checks(
            fake_voice,
            work_dir,
            require_daemon=False,
            skip_articulation_smoke=True,
            articulation_phrase=self.script.DEFAULT_ARTICULATION_PHRASE,
            articulation_expected_words=self.script.DEFAULT_ARTICULATION_EXPECTED_WORDS,
        )

    def test_smoke_checks_include_shared_cli_mcp_no_daemon_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_voice = self.make_fake_voice(tmp_path)

            checks = self.smoke_checks_with_fake_voice(fake_voice, tmp_path)

        by_name = {check["name"]: check for check in checks}
        self.assertTrue(by_name["stream_contract_no_daemon"]["ok"])
        self.assertEqual(
            by_name["stream_contract_no_daemon"]["contract"],
            "voice.webrtc_sidecar",
        )
        self.assertTrue(by_name["mcp_no_daemon_initializes"]["ok"])
        self.assertTrue(by_name["stream_no_daemon_fails_fast"]["ok"])
        self.assertFalse(by_name["daemon_detected"]["detected"])
        self.assertTrue(by_name["mcp_with_daemon_detects_daemon"]["skipped"])

    def test_smoke_checks_include_shared_cli_mcp_daemon_detection(self):
        old_value = os.environ.get("VOICE_FAKE_DAEMON")
        os.environ["VOICE_FAKE_DAEMON"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                fake_voice = self.make_fake_voice(tmp_path)

                checks = self.smoke_checks_with_fake_voice(fake_voice, tmp_path)
        finally:
            if old_value is None:
                os.environ.pop("VOICE_FAKE_DAEMON", None)
            else:
                os.environ["VOICE_FAKE_DAEMON"] = old_value

        by_name = {check["name"]: check for check in checks}
        self.assertTrue(by_name["daemon_detected"]["detected"])
        self.assertTrue(by_name["mcp_with_daemon_detects_daemon"]["ok"])
        self.assertTrue(by_name["stream_with_daemon_writes_pcm"]["ok"])


if __name__ == "__main__":
    unittest.main()
