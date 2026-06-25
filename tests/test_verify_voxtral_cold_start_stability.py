#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_voxtral_cold_start_stability.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_voxtral_cold_start_stability", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VoxtralColdStartStabilityVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script_module()

    def test_read_prompt_file_accepts_comments_bullets_and_numbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompts.txt"
            prompt_file.write_text(
                textwrap.dedent(
                    """\
                    # ignored
                    1. hello world
                    - Voxtral should pronounce its own made-up name clearly.

                    Read ticket A17, version 2.4.1, at 9:30 PM.
                    """
                ),
                encoding="utf-8",
            )

            prompts = self.script.read_prompt_file(prompt_file)

        self.assertEqual(
            prompts,
            [
                "hello world",
                "Voxtral should pronounce its own made-up name clearly.",
                "Read ticket A17, version 2.4.1, at 9:30 PM.",
            ],
        )

    def test_collect_prompts_defaults_to_single_prompt(self):
        prompts = self.script.collect_prompts(
            argparse.Namespace(suite=False, prompt_file=None, text=None)
        )

        self.assertEqual(prompts, [self.script.DEFAULT_TEXT])

    def test_collect_prompts_appends_suite_file_and_repeated_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "prompts.txt"
            prompt_file.write_text("1. from file\n", encoding="utf-8")

            prompts = self.script.collect_prompts(
                argparse.Namespace(
                    suite=True,
                    prompt_file=prompt_file,
                    text=["first explicit", "second explicit"],
                )
            )

        self.assertEqual(
            prompts[: len(self.script.DEFAULT_SUITE_TEXTS)],
            self.script.DEFAULT_SUITE_TEXTS,
        )
        self.assertEqual(prompts[-3:], ["from file", "first explicit", "second explicit"])

    def test_shape_and_failure_counters_are_prompt_aware(self):
        rows = [
            {
                "audio_duration_ms": 1733.333,
                "model_audio_duration_ms": 2080.0,
                "voxtral_audio_frames": 26,
                "voxtral_max_frames": 56,
                "ended": True,
            },
            {
                "audio_duration_ms": 3733.333,
                "model_audio_duration_ms": 4480.0,
                "voxtral_audio_frames": 56,
                "voxtral_max_frames": 56,
                "ended": False,
            },
            {
                "audio_duration_ms": 3000.0,
                "model_audio_duration_ms": 3600.0,
                "voxtral_audio_frames": 45,
                "voxtral_max_frames": None,
                "ended": False,
            },
        ]

        self.assertEqual(
            self.script.shape_key(rows[0]),
            (1733.333, 2080.0, 26, True),
        )
        self.assertEqual(self.script.did_not_end_count(rows), 2)
        self.assertEqual(self.script.frame_cap_hit_count(rows), 1)

    def test_bench_command_can_enable_auto_max_frames(self):
        command = self.script.bench_command(
            Path("/tmp/voice"),
            argparse.Namespace(
                voice="casual_male",
                speed=1.2,
                auto_max_frames=True,
            ),
            Path("/tmp/out"),
            "hello world",
        )

        self.assertIn("--voxtral-auto-max-frames", command)
        self.assertEqual(command[-1], "hello world")

    def test_prompt_slug_is_stable_and_has_fallback(self):
        self.assertEqual(
            self.script.prompt_slug(0, "Hello, world!"),
            "text1-hello-world",
        )
        self.assertEqual(
            self.script.prompt_slug(1, "!!!"),
            "text2-prompt",
        )

    def test_afinfo_summary_skips_when_afinfo_is_unavailable(self):
        with mock.patch.object(self.script.shutil, "which", return_value=None):
            summary = self.script.afinfo_summary(Path("/tmp/missing.wav"), timeout=1)

        self.assertIn("afinfo unavailable", summary)


if __name__ == "__main__":
    unittest.main()
