import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.evaluate import (
    build_parakeet_mlx_command,
    char_error_rate,
    edit_distance,
    normalize_text,
    parakeet_txt_output_path,
    score_pair,
    wav_duration_seconds,
    word_error_rate,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_normalize_text_lowercases_strips_punctuation_and_collapses_space(self):
        self.assertEqual(
            normalize_text("  Hello,   JSON-world!  "),
            "hello jsonworld",
        )

    def test_edit_distance_handles_insert_delete_and_substitute(self):
        self.assertEqual(edit_distance(["a", "b"], ["a", "b", "c"]), 1)
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "c"]), 1)
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "x", "c"]), 1)

    def test_word_error_rate_uses_levenshtein_not_position_mismatch(self):
        self.assertEqual(word_error_rate("alpha beta gamma", "alpha gamma"), 1 / 3)

    def test_score_pair_exact_match_has_zero_error_rates(self):
        score = score_pair("Hello, world!", "hello world")
        self.assertTrue(score["exact"])
        self.assertEqual(score["wer"], 0.0)
        self.assertEqual(score["cer"], 0.0)

    def test_empty_expected_and_actual_are_zero_error(self):
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(char_error_rate("", ""), 0.0)

    def test_empty_expected_with_actual_counts_insertions(self):
        self.assertEqual(word_error_rate("", "extra words"), 2.0)
        self.assertEqual(char_error_rate("", "abc"), 3.0)

    def test_wav_duration_supports_extensible_float_fixture_shape(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "extensible.wav"
            sample_rate = 24_000
            channels = 1
            bytes_per_sample = 4
            frames = 48_000
            block_align = channels * bytes_per_sample
            data_size = frames * block_align
            byte_rate = sample_rate * block_align
            fmt = (
                b"fmt "
                + (40).to_bytes(4, "little")
                + (0xFFFE).to_bytes(2, "little")
                + channels.to_bytes(2, "little")
                + sample_rate.to_bytes(4, "little")
                + byte_rate.to_bytes(4, "little")
                + block_align.to_bytes(2, "little")
                + (32).to_bytes(2, "little")
                + (22).to_bytes(2, "little")
                + (32).to_bytes(2, "little")
                + (1).to_bytes(4, "little")
                + (3).to_bytes(2, "little")
                + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
            )
            data = b"data" + data_size.to_bytes(4, "little") + (b"\x00" * data_size)
            riff_size = 4 + len(fmt) + len(data)
            path.write_bytes(b"RIFF" + riff_size.to_bytes(4, "little") + b"WAVE" + fmt + data)

            self.assertEqual(wav_duration_seconds(path), 2.0)

    def test_parakeet_output_path_uses_wav_stem(self):
        self.assertEqual(
            parakeet_txt_output_path(Path("/tmp/out"), Path("recordings/001.wav")),
            Path("/tmp/out/001.txt"),
        )

    def test_build_parakeet_mlx_command_uses_txt_output_and_optional_cache(self):
        command = build_parakeet_mlx_command(
            "parakeet-mlx",
            "mlx-community/parakeet-tdt-0.6b-v3",
            Path("eval/recordings/001.wav"),
            Path("/tmp/out"),
            Path("/tmp/hf-cache"),
        )

        self.assertEqual(command[0], "parakeet-mlx")
        self.assertIn("--output-format", command)
        self.assertIn("txt", command)
        self.assertIn("--output-template", command)
        self.assertIn("{filename}", command)
        self.assertIn("--cache-dir", command)
        self.assertIn("/tmp/hf-cache", command)


if __name__ == "__main__":
    unittest.main()
