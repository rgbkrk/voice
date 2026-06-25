#!/usr/bin/env python3
"""Verify repeated Voxtral cold-start bench runs keep stable audio shape.

This is a local runtime verifier for cold-start optimization work. It runs
`voice bench tts` in separate processes, writes WAV artifacts, and fails when
the same prompt/settings produce different frame counts, durations, or
termination state. That catches load-path changes that look faster but corrupt
or perturb generation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_TEXT = "A fast reply should arrive naturally."
DEFAULT_SUITE_TEXTS = [
    "hello world",
    DEFAULT_TEXT,
    "Voxtral should pronounce its own made-up name clearly.",
    "Please pause, then continue; do not add extra words.",
    "Read ticket A17, version 2.4.1, at 9:30 PM.",
    "If I ask a quick question, can you answer in one sentence?",
    "The voice should stay steady across a longer reply, "
    "even when the sentence reaches the realtime frame cap.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated no-playback Voxtral cold-start stability checks."
    )
    parser.add_argument(
        "--voice-bin",
        type=Path,
        default=None,
        help="voice binary to run; defaults to target/release/voice, then PATH",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="separate cold-start process count",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/voxtral-cold-start-stability"),
        help="directory for per-run JSON and WAV artifacts",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="prompt to verify; repeat for multiple prompts",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "file containing prompts, one per line; '#', '-' bullets, "
            "and '1.' numbering are accepted"
        ),
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="use the canonical varied Voxtral prompt suite",
    )
    parser.add_argument("--voice", default="casual_male")
    parser.add_argument("--speed", type=float, default=1.2)
    parser.add_argument(
        "--auto-max-frames",
        action="store_true",
        help="enable voice bench tts --voxtral-auto-max-frames",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--allow-shape-variants",
        action="store_true",
        help="report but do not fail when repeated runs produce different audio shapes",
    )
    parser.add_argument(
        "--skip-afinfo",
        action="store_true",
        help="skip macOS afinfo validation of saved WAV channel layout",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_voice_bin(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"voice binary is not executable: {path}")
        return path

    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return release_bin

    found = shutil.which("voice")
    if found:
        return Path(found).resolve()

    raise SystemExit("voice binary not found; build release voice or pass --voice-bin")


def run_command(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = timeout_text(exc.stdout)
        stderr = timeout_text(exc.stderr)
        if stderr:
            stderr += "\n"
        stderr += f"timed out after {exc.timeout}s"
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)


def timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def collect_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.suite:
        prompts.extend(DEFAULT_SUITE_TEXTS)
    if args.prompt_file is not None:
        prompts.extend(read_prompt_file(args.prompt_file))
    if args.text:
        prompts.extend(args.text)
    if not prompts:
        prompts.append(DEFAULT_TEXT)
    return prompts


def read_prompt_file(path: Path) -> list[str]:
    prompts: list[str] = []
    for raw_line in path.expanduser().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        numbered = line.split(".", 1)
        if len(numbered) == 2 and numbered[0].strip().isdigit():
            line = numbered[1].strip()
        if line:
            prompts.append(line)
    return prompts


def bench_command(
    voice_bin: Path,
    args: argparse.Namespace,
    run_dir: Path,
    text: str,
) -> list[str]:
    command = [
        str(voice_bin),
        "bench",
        "tts",
        "--quiet",
        "--engine",
        "voxtral",
        "--voxtral-voice",
        args.voice,
        "--runs",
        "1",
        "--voxtral-realtime",
        "--voxtral-kv-cache",
        "--voxtral-normalize-text",
        "--voxtral-pronunciation-aliases",
        "--speed",
        str(args.speed),
    ]
    if args.auto_max_frames:
        command.append("--voxtral-auto-max-frames")
    command.extend(
        [
            "--output-dir",
            str(run_dir),
            "--json",
            text,
        ]
    )
    return command


def read_bench(stdout: str, source: str, prompt_index: int, text: str) -> dict[str, Any]:
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{source}: bench output was not JSON: {exc}") from exc

    engines = report.get("engines")
    if not isinstance(engines, list) or len(engines) != 1:
        raise RuntimeError(f"{source}: expected exactly one engine report")
    engine = engines[0]
    runs = engine.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise RuntimeError(f"{source}: expected exactly one measured run")
    run = runs[0]
    output_wav = run.get("output_wav")
    if not output_wav:
        raise RuntimeError(f"{source}: bench did not report an output WAV")
    wav_path = Path(output_wav)
    if not wav_path.is_file():
        raise RuntimeError(f"{source}: output WAV is missing: {wav_path}")

    required_engine_fields = [
        "model_load_ms",
        "cold_first_code_frame_ms",
        "module_load_ms",
        "module_language_layers_load_ms",
    ]
    missing = [field for field in required_engine_fields if engine.get(field) is None]
    if missing:
        raise RuntimeError(f"{source}: missing load trace fields: {', '.join(missing)}")

    return {
        "source": source,
        "prompt_index": prompt_index,
        "text": text,
        "model_load_ms": engine["model_load_ms"],
        "module_load_ms": engine["module_load_ms"],
        "module_language_layers_load_ms": engine["module_language_layers_load_ms"],
        "cold_first_code_frame_ms": engine["cold_first_code_frame_ms"],
        "cold_first_audio_ms": engine["cold_first_audio_ms"],
        "first_code_frame_ms": run.get("first_code_frame_ms"),
        "first_audio_ms": run["first_audio_ms"],
        "total_ms": run["total_ms"],
        "audio_duration_ms": run["audio_duration_ms"],
        "model_audio_duration_ms": run.get("model_audio_duration_ms"),
        "voxtral_audio_frames": run.get("voxtral_audio_frames"),
        "voxtral_max_frames": run.get("voxtral_max_frames"),
        "voxtral_flow_steps": run.get("voxtral_flow_steps"),
        "voxtral_stream_begin_frames": run.get("voxtral_stream_begin_frames"),
        "ended": run.get("ended"),
        "output_wav": str(wav_path),
    }


def afinfo_summary(wav_path: Path, timeout: float) -> str:
    afinfo = shutil.which("afinfo")
    if not afinfo:
        raise RuntimeError("afinfo not found; pass --skip-afinfo on non-macOS hosts")
    result = run_command([afinfo, str(wav_path)], timeout)
    if result.returncode != 0:
        raise RuntimeError(f"afinfo failed for {wav_path}: {result.stderr.strip()}")
    summary_lines = []
    for line in result.stdout.splitlines():
        if any(
            key in line
            for key in ("Data format:", "Channel layout:", "estimated duration:")
        ):
            summary_lines.append(line.strip())
    summary = " | ".join(summary_lines)
    if "2 ch" not in result.stdout or "Stereo (L R)" not in result.stdout:
        raise RuntimeError(f"{wav_path}: expected stereo L/R WAV, got: {summary}")
    return summary


def shape_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["audio_duration_ms"],
        row["model_audio_duration_ms"],
        row["voxtral_audio_frames"],
        row["ended"],
    )


def did_not_end_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["ended"] is False)


def frame_cap_hit_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("voxtral_max_frames") is not None
        and row.get("voxtral_audio_frames") is not None
        and row["voxtral_audio_frames"] >= row["voxtral_max_frames"]
    )


def prompt_slug(index: int, text: str) -> str:
    slug = []
    for char in text.lower():
        if char.isalnum():
            slug.append(char)
        elif slug and slug[-1] != "-":
            slug.append("-")
        if len(slug) >= 42:
            break
    value = "".join(slug).strip("-")
    if not value:
        value = "prompt"
    return f"text{index + 1}-{value}"


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    voice_bin = resolve_voice_bin(args.voice_bin)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = collect_prompts(args)

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for prompt_index, text in enumerate(prompts):
        prompt_dir = output_dir / prompt_slug(prompt_index, text)
        for run_idx in range(1, args.runs + 1):
            run_dir = prompt_dir / f"run{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = bench_command(voice_bin, args, run_dir, text)
            result = run_command(command, args.timeout)
            bench_json = run_dir / "bench.json"
            bench_json.write_text(result.stdout, encoding="utf-8")
            (run_dir / "bench.stderr").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                failures.append(
                    f"prompt {prompt_index + 1} run {run_idx}: bench failed with "
                    f"{result.returncode}: {result.stderr.strip()}"
                )
                continue
            try:
                row = read_bench(
                    result.stdout,
                    f"prompt {prompt_index + 1} run {run_idx}",
                    prompt_index,
                    text,
                )
                if not args.skip_afinfo:
                    row["afinfo"] = afinfo_summary(Path(row["output_wav"]), args.timeout)
                rows.append(row)
            except RuntimeError as exc:
                failures.append(str(exc))

    if not rows:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1

    prompt_summaries = []
    for prompt_index, text in enumerate(prompts):
        prompt_rows = [row for row in rows if row["prompt_index"] == prompt_index]
        if not prompt_rows:
            failures.append(f"prompt {prompt_index + 1}: no successful runs")
            continue
        shape_variants = {shape_key(row) for row in prompt_rows}
        if len(shape_variants) > 1 and not args.allow_shape_variants:
            failures.append(
                f"prompt {prompt_index + 1}: audio shape varied across identical cold-start runs: "
                + ", ".join(str(variant) for variant in sorted(shape_variants))
            )
        prompt_summaries.append(
            {
                "prompt_index": prompt_index,
                "text": text,
                "runs_completed": len(prompt_rows),
                "shape_variants": [list(variant) for variant in sorted(shape_variants)],
                "did_not_end_count": did_not_end_count(prompt_rows),
                "frame_cap_hit_count": frame_cap_hit_count(prompt_rows),
                "model_load_ms": summarize([row["model_load_ms"] for row in prompt_rows]),
                "module_language_layers_load_ms": summarize(
                    [row["module_language_layers_load_ms"] for row in prompt_rows]
                ),
                "cold_first_code_frame_ms": summarize(
                    [row["cold_first_code_frame_ms"] for row in prompt_rows]
                ),
            }
        )

    summary = {
        "voice_bin": str(voice_bin),
        "output_dir": str(output_dir),
        "prompts_requested": len(prompts),
        "runs_requested": args.runs,
        "runs_completed": len(rows),
        "did_not_end_count": did_not_end_count(rows),
        "frame_cap_hit_count": frame_cap_hit_count(rows),
        "prompt_summaries": prompt_summaries,
        "model_load_ms": summarize([row["model_load_ms"] for row in rows]),
        "module_language_layers_load_ms": summarize(
            [row["module_language_layers_load_ms"] for row in rows]
        ),
        "cold_first_code_frame_ms": summarize(
            [row["cold_first_code_frame_ms"] for row in rows]
        ),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    return 1 if failures else 0


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


if __name__ == "__main__":
    raise SystemExit(main())
