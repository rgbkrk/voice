#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main()
