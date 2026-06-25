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
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default="casual_male")
    parser.add_argument("--speed", type=float, default=1.2)
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


def bench_command(voice_bin: Path, args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
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
        "--output-dir",
        str(run_dir),
        "--json",
        args.text,
    ]


def read_bench(stdout: str, source: str) -> dict[str, Any]:
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
        if any(key in line for key in ("Data format:", "Channel layout:", "estimated duration:")):
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


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    voice_bin = resolve_voice_bin(args.voice_bin)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for run_idx in range(1, args.runs + 1):
        run_dir = output_dir / f"run{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = bench_command(voice_bin, args, run_dir)
        result = run_command(command, args.timeout)
        bench_json = run_dir / "bench.json"
        bench_json.write_text(result.stdout, encoding="utf-8")
        (run_dir / "bench.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            failures.append(
                f"run {run_idx}: bench failed with {result.returncode}: {result.stderr.strip()}"
            )
            continue
        try:
            row = read_bench(result.stdout, f"run {run_idx}")
            if not args.skip_afinfo:
                row["afinfo"] = afinfo_summary(Path(row["output_wav"]), args.timeout)
            rows.append(row)
        except RuntimeError as exc:
            failures.append(str(exc))

    if not rows:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1

    shape_variants = {shape_key(row) for row in rows}
    if len(shape_variants) > 1 and not args.allow_shape_variants:
        failures.append(
            "audio shape varied across identical cold-start runs: "
            + ", ".join(str(variant) for variant in sorted(shape_variants))
        )

    summary = {
        "voice_bin": str(voice_bin),
        "output_dir": str(output_dir),
        "runs_requested": args.runs,
        "runs_completed": len(rows),
        "shape_variants": [list(variant) for variant in sorted(shape_variants)],
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
