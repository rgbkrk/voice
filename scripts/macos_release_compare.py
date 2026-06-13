#!/usr/bin/env python3
"""Compare the latest stable voice release with the current checkout on macOS.

The script intentionally forces `VOICE_DAEMON_SOCKET` to a missing path for the
core benchmarks so it measures the plain CLI/local model path instead of an
already-running daemon. It also runs separate smoke checks for daemon-backed
streaming when a daemon is available.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import wave
from pathlib import Path


DEFAULT_PHRASES = [
    "Hermes is checking whether voice synthesis still starts quickly.",
    "The nteract runtime should pronounce technical identifiers clearly.",
    "Streaming audio frames need predictable timing for WebRTC clients.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current voice performance against the latest stable macOS release."
    )
    parser.add_argument("--repo", default="rgbkrk/voice", help="GitHub repo to download from")
    parser.add_argument("--release", help="Release tag to use as the old baseline")
    parser.add_argument("--old-voice", type=Path, help="Path to an existing old voice binary")
    parser.add_argument("--new-voice", type=Path, help="Path to an existing new voice binary")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not build target/release/voice when --new-voice is omitted",
    )
    parser.add_argument("--iters", type=int, default=3, help="TTS iterations per phrase")
    parser.add_argument("--stt-iters", type=int, default=1, help="STT iterations per WAV file")
    parser.add_argument(
        "--stt-recordings",
        type=Path,
        default=Path("eval/recordings"),
        help="Directory of WAV files for optional STT timing",
    )
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT timing")
    parser.add_argument(
        "--require-daemon",
        action="store_true",
        help="Fail if a daemon is not available for the daemon-backed stream smoke check",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.20,
        help="Fail if new mean elapsed time exceeds old mean by this multiplier",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for downloaded releases, generated WAVs, and JSON output",
    )
    parser.add_argument("--keep", action="store_true", help="Keep temporary work directory")
    parser.add_argument("--json-out", type=Path, help="Write machine-readable results here")
    return parser.parse_args()


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    capture: bool = False,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
        check=check,
    )


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"required tool not found on PATH: {name}")


def resolve_new_voice(args: argparse.Namespace) -> Path:
    if args.new_voice:
        return args.new_voice.resolve()
    if args.skip_build:
        raise SystemExit("--skip-build requires --new-voice")

    print("Building current checkout: cargo build --release -p voice")
    run(["cargo", "build", "--release", "-p", "voice"])
    return Path("target/release/voice").resolve()


def resolve_old_voice(args: argparse.Namespace, work_dir: Path) -> tuple[Path, str]:
    if args.old_voice:
        return args.old_voice.resolve(), "custom-old"

    require_tool("gh")
    old_dir = work_dir / "old-release"
    old_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gh",
        "release",
        "download",
        "--repo",
        args.repo,
        "--pattern",
        "voice-aarch64-apple-darwin.tar.gz",
        "--dir",
        str(old_dir),
        "--clobber",
    ]
    if args.release:
        cmd.insert(3, args.release)

    label = args.release or "latest-stable-release"
    print(f"Downloading old baseline from {args.repo} ({label})")
    run(cmd)

    archives = sorted(old_dir.glob("voice-aarch64-apple-darwin.tar.gz"))
    if not archives:
        raise SystemExit(f"release asset not found in {old_dir}")

    extract_dir = old_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(archives[0], "r:gz") as tar:
        tar.extractall(extract_dir)

    voice = extract_dir / "voice"
    if not voice.exists():
        raise SystemExit(f"release archive did not contain a voice binary: {archives[0]}")
    voice.chmod(0o755)
    return voice.resolve(), label


def wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as reader:
            frames = reader.getnframes()
            rate = reader.getframerate()
        return frames / rate
    except (wave.Error, OSError, EOFError):
        duration = riff_wav_duration_seconds(path)
        if duration is None:
            raise
        return duration


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


def missing_daemon_env(work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["VOICE_DAEMON_SOCKET"] = str(work_dir / "no-daemon.sock")
    return env


def time_command(cmd: list[str], *, env: dict[str, str], timeout: int | None = None) -> float:
    started = time.perf_counter()
    run(cmd, env=env, timeout=timeout)
    return time.perf_counter() - started


def benchmark_tts(
    label: str,
    voice: Path,
    work_dir: Path,
    iterations: int,
) -> list[dict[str, object]]:
    env = missing_daemon_env(work_dir)
    out_dir = work_dir / "tts" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup = out_dir / "warmup.wav"
    run([str(voice), "say", "-q", "-o", str(warmup), "Warm up the voice model."], env=env)

    rows: list[dict[str, object]] = []
    for iteration in range(1, iterations + 1):
        for phrase_index, phrase in enumerate(DEFAULT_PHRASES, start=1):
            wav = out_dir / f"{iteration:02d}-{phrase_index:02d}.wav"
            elapsed = time_command(
                [str(voice), "say", "-q", "-o", str(wav), phrase],
                env=env,
                timeout=240,
            )
            duration = wav_duration_seconds(wav)
            rows.append(
                {
                    "kind": "tts",
                    "label": label,
                    "iteration": iteration,
                    "phrase_index": phrase_index,
                    "elapsed_seconds": elapsed,
                    "audio_seconds": duration,
                    "rtf": elapsed / duration if duration else None,
                    "output": str(wav),
                }
            )
    return rows


def benchmark_stt(
    label: str,
    voice: Path,
    recordings: list[Path],
    work_dir: Path,
    iterations: int,
) -> list[dict[str, object]]:
    env = missing_daemon_env(work_dir)
    rows: list[dict[str, object]] = []
    if not recordings:
        return rows

    run([str(voice), "transcribe", "-q", str(recordings[0])], env=env, capture=True, timeout=300)
    for iteration in range(1, iterations + 1):
        for wav in recordings:
            started = time.perf_counter()
            result = run(
                [str(voice), "transcribe", "-q", str(wav)],
                env=env,
                capture=True,
                timeout=300,
            )
            elapsed = time.perf_counter() - started
            duration = wav_duration_seconds(wav)
            rows.append(
                {
                    "kind": "stt",
                    "label": label,
                    "iteration": iteration,
                    "recording": str(wav),
                    "elapsed_seconds": elapsed,
                    "audio_seconds": duration,
                    "rtf": elapsed / duration if duration else None,
                    "transcript": result.stdout.strip(),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]], kind: str, label: str) -> dict[str, float | int]:
    values = [
        float(row["elapsed_seconds"])
        for row in rows
        if row["kind"] == kind and row["label"] == label
    ]
    rtfs = [
        float(row["rtf"])
        for row in rows
        if row["kind"] == kind and row["label"] == label and row["rtf"] is not None
    ]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_elapsed_seconds": statistics.fmean(values),
        "median_elapsed_seconds": statistics.median(values),
        "mean_rtf": statistics.fmean(rtfs) if rtfs else 0.0,
        "median_rtf": statistics.median(rtfs) if rtfs else 0.0,
    }


def compare_summary(
    summaries: dict[str, dict[str, float | int]],
    kind: str,
    threshold: float,
) -> tuple[str, bool]:
    old = summaries.get(f"{kind}:old", {})
    new = summaries.get(f"{kind}:new", {})
    if not old.get("count") or not new.get("count"):
        return f"{kind}: skipped", True

    old_mean = float(old["mean_elapsed_seconds"])
    new_mean = float(new["mean_elapsed_seconds"])
    ratio = new_mean / old_mean if old_mean else float("inf")
    ok = ratio <= threshold
    status = "ok" if ok else "regression"
    return f"{kind}: {status}, new/old mean elapsed = {ratio:.2f}x", ok


def smoke_checks(
    new_voice: Path, work_dir: Path, *, require_daemon: bool
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    env = missing_daemon_env(work_dir)

    no_daemon_wav = work_dir / "smoke-no-daemon.wav"
    run(
        [
            str(new_voice),
            "say",
            "-q",
            "-o",
            str(no_daemon_wav),
            "No daemon file synthesis smoke test.",
        ],
        env=env,
        timeout=240,
    )
    checks.append(
        {
            "name": "say_no_daemon_writes_wav",
            "ok": no_daemon_wav.exists() and no_daemon_wav.stat().st_size > 44,
            "path": str(no_daemon_wav),
        }
    )

    mcp_input = (
        '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}\n'
        '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}\n'
    )
    mcp = run(
        [str(new_voice), "mcp", "-q"],
        env=env,
        input_text=mcp_input,
        capture=True,
        timeout=240,
    )
    checks.append(
        {
            "name": "mcp_no_daemon_initializes",
            "ok": '"serverInfo"' in mcp.stdout and '"tools"' in mcp.stdout,
        }
    )

    stream = run(
        [str(new_voice), "stream", "-o", str(work_dir / "no-daemon-stream.pcm"), "No daemon."],
        env=env,
        capture=True,
        check=False,
        timeout=30,
    )
    checks.append(
        {
            "name": "stream_no_daemon_fails_fast",
            "ok": stream.returncode != 0,
            "note": "voice stream is daemon-backed only",
        }
    )

    daemon = subprocess.run(
        [str(new_voice), "daemon", "status", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    daemon_available = daemon.returncode == 0
    checks.append(
        {
            "name": "daemon_detected",
            "ok": daemon_available or not require_daemon,
            "detected": daemon_available,
            "note": "daemon stream check runs only when a daemon is detected",
        }
    )

    if daemon_available:
        pcm = work_dir / "stream-daemon.pcm"
        run(
            [str(new_voice), "stream", "-o", str(pcm), "Daemon streaming smoke test."],
            capture=True,
            timeout=240,
        )
        checks.append(
            {
                "name": "stream_with_daemon_writes_pcm",
                "ok": pcm.exists() and pcm.stat().st_size > 0,
                "path": str(pcm),
            }
        )
    else:
        checks.append(
            {
                "name": "stream_with_daemon_writes_pcm",
                "ok": not require_daemon,
                "skipped": True,
                "note": "daemon not detected",
            }
        )
    return checks


def main() -> int:
    args = parse_args()
    if args.iters < 1 or args.stt_iters < 1:
        raise SystemExit("--iters and --stt-iters must be positive")

    if platform.system() != "Darwin":
        print("warning: this script is intended for macOS/Apple Silicon", file=sys.stderr)

    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="voice-release-compare-")
        work_dir = Path(temp_ctx.name)

    print(f"Work dir: {work_dir}")
    old_voice, old_label = resolve_old_voice(args, work_dir)
    new_voice = resolve_new_voice(args)

    print(f"Old voice ({old_label}): {old_voice}")
    print(f"New voice: {new_voice}")

    all_rows: list[dict[str, object]] = []
    all_rows.extend(benchmark_tts("old", old_voice, work_dir, args.iters))
    all_rows.extend(benchmark_tts("new", new_voice, work_dir, args.iters))

    recordings = sorted(args.stt_recordings.glob("*.wav")) if args.stt_recordings.exists() else []
    if args.skip_stt:
        recordings = []
    if recordings:
        print(f"Running STT comparison with {len(recordings)} recordings from {args.stt_recordings}")
        all_rows.extend(benchmark_stt("old", old_voice, recordings, work_dir, args.stt_iters))
        all_rows.extend(benchmark_stt("new", new_voice, recordings, work_dir, args.stt_iters))
    else:
        print("Skipping STT comparison: no WAV fixtures found or --skip-stt was set")

    summaries = {
        "tts:old": summarize(all_rows, "tts", "old"),
        "tts:new": summarize(all_rows, "tts", "new"),
        "stt:old": summarize(all_rows, "stt", "old"),
        "stt:new": summarize(all_rows, "stt", "new"),
    }

    checks = smoke_checks(new_voice, work_dir, require_daemon=args.require_daemon)
    comparisons = []
    ok = True
    for kind in ["tts", "stt"]:
        message, comparison_ok = compare_summary(summaries, kind, args.threshold)
        comparisons.append({"kind": kind, "message": message, "ok": comparison_ok})
        ok = ok and comparison_ok
        print(message)

    for check in checks:
        status = "ok" if check["ok"] else "failed"
        print(f"{check['name']}: {status}")
        ok = ok and bool(check["ok"])

    result = {
        "old_voice": str(old_voice),
        "new_voice": str(new_voice),
        "threshold": args.threshold,
        "summaries": summaries,
        "comparisons": comparisons,
        "smoke_checks": checks,
        "rows": all_rows,
    }

    json_out = args.json_out or (work_dir / "release-compare.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote JSON results to {json_out}")

    if args.keep or args.work_dir:
        print(f"Kept work dir: {work_dir}")
    elif temp_ctx is not None:
        temp_ctx.cleanup()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
