#!/usr/bin/env python3
"""Evaluate voice STT output with WER/CER, timing, and JSON output."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

DEFAULT_VOICE_MODEL = "distil-whisper/distil-medium.en"
DEFAULT_PARAKEET_MLX_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def normalize_text(text: str) -> str:
    lowered = text.lower()
    alnum_space = re.sub(r"[^a-z0-9\s]", "", lowered)
    return " ".join(alnum_space.split())


def edit_distance(expected: Sequence[str], actual: Sequence[str]) -> int:
    if not expected:
        return len(actual)
    if not actual:
        return len(expected)

    previous = list(range(len(actual) + 1))
    for i, expected_item in enumerate(expected, start=1):
        current = [i]
        for j, actual_item in enumerate(actual, start=1):
            cost = 0 if expected_item == actual_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _error_rate(errors: int, expected_count: int, actual_count: int) -> float:
    if expected_count == 0:
        return 0.0 if actual_count == 0 else float(errors)
    return errors / expected_count


def word_error_rate(expected: str, actual: str) -> float:
    expected_words = normalize_text(expected).split()
    actual_words = normalize_text(actual).split()
    errors = edit_distance(expected_words, actual_words)
    return _error_rate(errors, len(expected_words), len(actual_words))


def char_error_rate(expected: str, actual: str) -> float:
    expected_chars = list(normalize_text(expected).replace(" ", ""))
    actual_chars = list(normalize_text(actual).replace(" ", ""))
    errors = edit_distance(expected_chars, actual_chars)
    return _error_rate(errors, len(expected_chars), len(actual_chars))


def score_pair(expected: str, actual: str) -> dict[str, object]:
    expected_normalized = normalize_text(expected)
    actual_normalized = normalize_text(actual)
    expected_words = expected_normalized.split()
    actual_words = actual_normalized.split()
    expected_chars = list(expected_normalized.replace(" ", ""))
    actual_chars = list(actual_normalized.replace(" ", ""))
    word_errors = edit_distance(expected_words, actual_words)
    char_errors = edit_distance(expected_chars, actual_chars)

    return {
        "expected_normalized": expected_normalized,
        "actual_normalized": actual_normalized,
        "exact": expected_normalized == actual_normalized,
        "word_errors": word_errors,
        "word_count": len(expected_words),
        "wer": _error_rate(word_errors, len(expected_words), len(actual_words)),
        "char_errors": char_errors,
        "char_count": len(expected_chars),
        "cer": _error_rate(char_errors, len(expected_chars), len(actual_chars)),
    }


def wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as reader:
            frame_rate = reader.getframerate()
            if frame_rate == 0:
                return None
            return reader.getnframes() / frame_rate
    except (wave.Error, OSError, EOFError):
        return riff_wav_duration_seconds(path)


def riff_wav_duration_seconds(path: Path) -> float | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None

    sample_rate = None
    block_align = None
    data_size = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = min(chunk_start + chunk_size, len(data))
        chunk = data[chunk_start:chunk_end]

        if chunk_id == b"fmt " and len(chunk) >= 16:
            _format_tag, _channels, sample_rate, _byte_rate, block_align, _bits = (
                struct.unpack_from("<HHIIHH", chunk, 0)
            )
        elif chunk_id == b"data":
            data_size = chunk_size

        offset = chunk_start + chunk_size + (chunk_size % 2)

    if not sample_rate or not block_align or data_size is None:
        return None
    return (data_size / block_align) / sample_rate


def recording_pairs(recordings: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for text_path in sorted(recordings.glob("*.txt")):
        wav_path = text_path.with_suffix(".wav")
        if wav_path.exists():
            pairs.append((text_path.stem, text_path, wav_path))
    return pairs


def transcribe_voice(voice: str, model: str, wav_path: Path) -> tuple[str, float, str | None]:
    env = os.environ.copy()
    env["STT_MODEL"] = model
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [voice, "transcribe", "-q", str(wav_path)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as err:
        return "", time.perf_counter() - start, str(err)

    elapsed = time.perf_counter() - start
    transcript = completed.stdout.strip()
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"exit status {completed.returncode}"
        return transcript, elapsed, message
    return transcript, elapsed, None


def parakeet_txt_output_path(output_dir: Path, wav_path: Path) -> Path:
    return output_dir / f"{wav_path.stem}.txt"


def build_parakeet_mlx_command(
    parakeet_bin: str,
    model: str,
    wav_path: Path,
    output_dir: Path,
    cache_dir: Path | None,
) -> list[str]:
    command = [
        parakeet_bin,
        str(wav_path),
        "--model",
        model,
        "--output-dir",
        str(output_dir),
        "--output-format",
        "txt",
        "--output-template",
        "{filename}",
    ]
    if cache_dir is not None:
        command.extend(["--cache-dir", str(cache_dir)])
    return command


def transcribe_parakeet_mlx(
    parakeet_bin: str,
    model: str,
    wav_path: Path,
    cache_dir: Path | None,
) -> tuple[str, float, str | None]:
    start = time.perf_counter()
    with TemporaryDirectory(prefix="voice-parakeet-mlx-") as tmpdir:
        output_dir = Path(tmpdir)
        command = build_parakeet_mlx_command(
            parakeet_bin,
            model,
            wav_path,
            output_dir,
            cache_dir,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as err:
            return "", time.perf_counter() - start, str(err)

        elapsed = time.perf_counter() - start
        output_path = parakeet_txt_output_path(output_dir, wav_path)
        transcript = ""
        if output_path.exists():
            transcript = output_path.read_text(encoding="utf-8").strip()
        if completed.returncode != 0:
            message = completed.stderr.strip() or f"exit status {completed.returncode}"
            return transcript, elapsed, message
        if not output_path.exists():
            message = f"expected Parakeet MLX output not found: {output_path}"
            return transcript, elapsed, message
        return transcript, elapsed, None


def transcribe(
    engine: str,
    voice: str,
    parakeet_bin: str,
    model: str,
    wav_path: Path,
    parakeet_cache_dir: Path | None,
) -> tuple[str, float, str | None]:
    if engine == "voice":
        return transcribe_voice(voice, model, wav_path)
    if engine == "parakeet-mlx":
        return transcribe_parakeet_mlx(parakeet_bin, model, wav_path, parakeet_cache_dir)
    raise ValueError(f"unsupported engine: {engine}")


def evaluate(
    recordings: Path,
    engine: str,
    voice: str,
    parakeet_bin: str,
    model: str,
    parakeet_cache_dir: Path | None,
) -> dict[str, object]:
    items = []
    for item_id, text_path, wav_path in recording_pairs(recordings):
        expected = text_path.read_text(encoding="utf-8").strip()
        transcript, elapsed_seconds, error = transcribe(
            engine,
            voice,
            parakeet_bin,
            model,
            wav_path,
            parakeet_cache_dir,
        )
        duration_seconds = wav_duration_seconds(wav_path)
        score = score_pair(expected, transcript)
        item = {
            "id": item_id,
            "text_path": str(text_path),
            "wav_path": str(wav_path),
            "expected": expected,
            "actual": transcript,
            "elapsed_seconds": elapsed_seconds,
            "duration_seconds": duration_seconds,
            **score,
        }
        if error is not None:
            item["error"] = error
        items.append(item)

    total = len(items)
    exact = sum(1 for item in items if item["exact"])
    total_audio_seconds = sum(
        item["duration_seconds"] or 0.0 for item in items
    )
    total_elapsed_seconds = sum(float(item["elapsed_seconds"]) for item in items)
    mean_wer = (
        sum(float(item["wer"]) for item in items) / total if total else 0.0
    )
    mean_cer = (
        sum(float(item["cer"]) for item in items) / total if total else 0.0
    )

    return {
        "engine": engine,
        "model": model,
        "voice_binary": voice,
        "parakeet_binary": parakeet_bin,
        "parakeet_cache_dir": str(parakeet_cache_dir) if parakeet_cache_dir else None,
        "recordings": str(recordings),
        "total": total,
        "exact": exact,
        "mean_wer": mean_wer,
        "mean_cer": mean_cer,
        "total_audio_seconds": total_audio_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "rtf": (
            total_elapsed_seconds / total_audio_seconds
            if total_audio_seconds > 0.0
            else None
        ),
        "items": items,
    }


def print_summary(summary: dict[str, object]) -> None:
    total = int(summary["total"])
    exact = int(summary["exact"])
    mean_wer = float(summary["mean_wer"])
    mean_cer = float(summary["mean_cer"])
    rtf = summary["rtf"]

    print(f"=== Engine: {summary['engine']} | Model: {summary['model']} ===")
    for item in summary["items"]:
        status = "PASS" if item["exact"] else f"WER={float(item['wer']) * 100:.1f}%"
        print(f"  {item['id']} [{status}]")
        if not item["exact"]:
            print(f"    expected: {item['expected_normalized']}")
            print(f"    got:      {item['actual_normalized']}")
        if "error" in item:
            print(f"    error:    {item['error']}")

    rtf_text = "n/a" if rtf is None else f"{float(rtf):.2f}x"
    print()
    print(
        f"  Exact: {exact}/{total}; "
        f"WER: {mean_wer * 100:.1f}%; "
        f"CER: {mean_cer * 100:.1f}%; "
        f"RTF: {rtf_text}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", required=True, type=Path)
    parser.add_argument(
        "--engine",
        choices=["voice", "parakeet-mlx"],
        default="voice",
        help="Transcription engine to evaluate.",
    )
    parser.add_argument("--voice", default="./target/release/voice")
    parser.add_argument(
        "--parakeet-bin",
        default="parakeet-mlx",
        help="Parakeet MLX CLI path when --engine parakeet-mlx is used.",
    )
    parser.add_argument(
        "--parakeet-cache-dir",
        type=Path,
        help="Optional HuggingFace cache directory for Parakeet MLX models.",
    )
    parser.add_argument("--model")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.recordings.is_dir():
        parser.error(f"{args.recordings} is not a directory")

    model = args.model
    if model is None:
        model = (
            DEFAULT_PARAKEET_MLX_MODEL
            if args.engine == "parakeet-mlx"
            else DEFAULT_VOICE_MODEL
        )

    summary = evaluate(
        args.recordings,
        args.engine,
        args.voice,
        args.parakeet_bin,
        model,
        args.parakeet_cache_dir,
    )
    print_summary(summary)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
