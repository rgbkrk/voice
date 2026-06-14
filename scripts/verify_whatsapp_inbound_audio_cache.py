#!/usr/bin/env python3
"""Verify cached inbound WhatsApp audio downloaded by the local bridge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def default_voice_bin() -> str:
    env_value = os.environ.get("VOICE_BIN")
    if env_value:
        return env_value
    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return str(release_bin)
    found = shutil.which("voice")
    return found or "voice"


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        return str(path.resolve())
    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return str(Path(found).resolve())


def run_command(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def discover_audio_files(audio_cache_dir: Path) -> list[Path]:
    if not audio_cache_dir.is_dir():
        return []
    files = [
        path
        for path in audio_cache_dir.iterdir()
        if path.is_file()
        and path.name.startswith("aud_")
        and path.suffix.lower() in {".ogg", ".opus", ".m4a"}
    ]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def audio_file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def audio_file_snapshot(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in paths:
        signature = audio_file_signature(path)
        if signature is not None:
            snapshot[path.resolve()] = signature
    return snapshot


def fresh_audio_files(
    paths: list[Path],
    *,
    baseline: dict[Path, tuple[int, int]],
) -> list[Path]:
    fresh: list[Path] = []
    for path in paths:
        signature = audio_file_signature(path)
        if signature is None:
            continue
        if baseline.get(path.resolve()) != signature:
            fresh.append(path)
    return fresh


def wait_for_fresh_audio_files(
    audio_cache_dir: Path,
    *,
    baseline: dict[Path, tuple[int, int]],
    wait_seconds: float,
) -> tuple[list[Path], list[Path]]:
    deadline = time.monotonic() + wait_seconds
    discovered = discover_audio_files(audio_cache_dir)
    fresh = fresh_audio_files(discovered, baseline=baseline)
    while not fresh and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.1, deadline - time.monotonic())))
        discovered = discover_audio_files(audio_cache_dir)
        fresh = fresh_audio_files(discovered, baseline=baseline)
    return fresh, discovered


def probe_audio(path: Path, *, timeout: float, skip_ffprobe: bool) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    probe: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "magic": None,
        "ffprobe": None,
    }
    if not path.is_file():
        return probe, [f"inbound audio file not found: {path}"]
    if not path.name.startswith("aud_"):
        failures.append(
            f"inbound audio file name must start with aud_ to match bridge downloads: {path.name}"
        )
    if probe["size_bytes"] <= 64:
        failures.append(f"inbound audio file is too small: {probe['size_bytes']} bytes")

    with path.open("rb") as handle:
        probe["magic"] = handle.read(4).decode("ascii", errors="replace")

    ffprobe = shutil.which("ffprobe")
    if skip_ffprobe or not ffprobe:
        probe["ffprobe"] = {"skipped": True, "reason": "ffprobe unavailable or skipped"}
        return probe, failures

    completed = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        timeout=timeout,
    )
    probe["ffprobe"] = {
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        failures.append(f"ffprobe failed for {path}: {completed.stderr.strip()}")
        return probe, failures

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        failures.append(f"ffprobe did not return JSON for {path}: {exc}")
        return probe, failures

    stream = (payload.get("streams") or [{}])[0]
    probe["ffprobe"]["stream"] = stream
    codec = str(stream.get("codec_name") or "")
    if not codec:
        failures.append(f"ffprobe found no audio codec for {path}")
    channels = int(stream.get("channels") or 0)
    if channels <= 0:
        failures.append(f"ffprobe found no audio channels for {path}")
    return probe, failures


def transcribe_audio(
    voice_bin: str,
    path: Path,
    *,
    timeout: float,
) -> tuple[dict[str, Any], list[str]]:
    command = [voice_bin, "--quiet", "stream-transcribe", "--json", str(path)]
    completed = run_command(command, timeout=timeout)
    result: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "events": [],
        "terminal_event": None,
    }
    failures: list[str] = []
    if completed.returncode != 0:
        failures.append(f"voice stream-transcribe failed for {path}: {completed.stderr.strip()}")
        return result, failures

    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"stream-transcribe emitted non-JSON line: {exc}")
            continue
        result["events"].append(event)
        if event.get("event") in {"stt.transcribed", "stt.error"}:
            result["terminal_event"] = event

    terminal = result.get("terminal_event")
    if not terminal:
        failures.append("stream-transcribe did not emit a terminal STT event")
    elif terminal.get("event") != "stt.transcribed":
        failures.append(f"stream-transcribe terminal event was {terminal.get('event')!r}")
    else:
        data = terminal.get("data") or {}
        if not str(data.get("text") or "").strip():
            failures.append("stream-transcribe produced an empty transcript")
        if int(data.get("frames") or 0) <= 0:
            failures.append("stream-transcribe reported no audio frames")
        if int(data.get("audio_duration_ms") or 0) <= 0:
            failures.append("stream-transcribe reported no audio duration")
    return result, failures


def verify(args: argparse.Namespace) -> dict[str, Any]:
    hermes_home = args.hermes_home.expanduser().resolve()
    audio_cache_dir = (
        args.audio_cache_dir.expanduser().resolve()
        if args.audio_cache_dir
        else hermes_home / "audio_cache"
    )
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")

    failures: list[str] = []
    warnings: list[str] = []
    if args.wait_fresh_seconds < 0:
        failures.append("--wait-fresh-seconds must be non-negative")
    if args.require_fresh_audio and args.wait_fresh_seconds <= 0:
        failures.append("--require-fresh-audio requires --wait-fresh-seconds")

    selected_files: list[Path]
    discovered = discover_audio_files(audio_cache_dir)
    baseline = audio_file_snapshot(discovered)
    fresh_files: list[Path] = []

    if args.wait_fresh_seconds > 0 and not failures:
        fresh_files, discovered = wait_for_fresh_audio_files(
            audio_cache_dir,
            baseline=baseline,
            wait_seconds=args.wait_fresh_seconds,
        )

    if args.audio_file:
        selected_files = [path.expanduser().resolve() for path in args.audio_file]
    elif args.wait_fresh_seconds > 0:
        if fresh_files or args.require_fresh_audio:
            selected_files = fresh_files[: args.max_files]
        else:
            selected_files = discovered[: args.max_files]
    else:
        selected_files = discovered[: args.max_files]

    if not selected_files:
        if args.wait_fresh_seconds > 0 and args.require_fresh_audio:
            message = (
                "no fresh bridge-downloaded inbound audio files found in "
                f"{audio_cache_dir} during {args.wait_fresh_seconds}s watch"
            )
        else:
            message = f"no bridge-downloaded inbound audio files found in {audio_cache_dir}"
        if args.require_cache or args.require_fresh_audio:
            failures.append(message)
        else:
            warnings.append(message)

    checks: dict[str, Any] = {
        "hermes_home": str(hermes_home),
        "audio_cache_dir": str(audio_cache_dir),
        "voice_bin": voice_bin,
        "discovered_count": len(discovered),
        "selected_files": [str(path) for path in selected_files],
        "fresh_watch": {
            "wait_seconds": args.wait_fresh_seconds,
            "drains_bridge_messages": False,
            "baseline_count": len(baseline),
            "fresh_count": len(fresh_files),
            "fresh_files": [str(path) for path in fresh_files],
            "skipped": args.wait_fresh_seconds <= 0,
        },
        "audio": [],
    }

    for path in selected_files:
        audio_check: dict[str, Any] = {"path": str(path)}
        probe, probe_failures = probe_audio(
            path,
            timeout=args.timeout,
            skip_ffprobe=args.skip_ffprobe,
        )
        audio_check["probe"] = probe
        failures.extend(probe_failures)

        if args.run_stt and not probe_failures:
            stt, stt_failures = transcribe_audio(
                voice_bin,
                path,
                timeout=args.stt_timeout,
            )
            audio_check["stt"] = stt
            failures.extend(stt_failures)
        else:
            audio_check["stt"] = {
                "skipped": True,
                "reason": "pass --run-stt" if not args.run_stt else "probe failed",
            }
        checks["audio"].append(audio_check)

    return {
        "success": not failures,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=default_voice_bin())
    parser.add_argument("--hermes-home", type=Path, default=default_hermes_home())
    parser.add_argument("--audio-cache-dir", type=Path, default=None)
    parser.add_argument("--audio-file", type=Path, action="append", default=None)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument(
        "--wait-fresh-seconds",
        type=float,
        default=0.0,
        help="watch the audio cache for a new or updated aud_* file without polling the bridge",
    )
    parser.add_argument(
        "--require-fresh-audio",
        action="store_true",
        help="fail unless --wait-fresh-seconds observes a fresh inbound audio artifact",
    )
    parser.add_argument("--run-stt", action="store_true")
    parser.add_argument("--skip-ffprobe", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--stt-timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    return parser


def human_summary(result: dict[str, Any]) -> None:
    checks = result["checks"]
    if result["success"]:
        print("ok: WhatsApp inbound audio cache verifier passed")
    else:
        print("error: WhatsApp inbound audio cache verifier failed", file=sys.stderr)
        for failure in result["failures"]:
            print(f"- {failure}", file=sys.stderr)

    print(f"audio_cache_dir={checks['audio_cache_dir']}")
    print(f"discovered_count={checks['discovered_count']}")
    fresh = checks.get("fresh_watch") or {}
    if not fresh.get("skipped"):
        print(
            "fresh_watch="
            f"fresh_count={fresh.get('fresh_count')} "
            f"wait_seconds={fresh.get('wait_seconds')} "
            f"drains_messages={fresh.get('drains_bridge_messages')}"
        )
    for index, audio in enumerate(checks["audio"], start=1):
        probe = audio.get("probe") or {}
        stream = ((probe.get("ffprobe") or {}).get("stream") or {})
        print(
            f"audio[{index}]={probe.get('path')} "
            f"size={probe.get('size_bytes')} "
            f"codec={stream.get('codec_name') or '<unknown>'} "
            f"sample_rate={stream.get('sample_rate') or '<unknown>'} "
            f"channels={stream.get('channels') or '<unknown>'}"
        )
        stt = audio.get("stt") or {}
        if stt.get("skipped"):
            print(f"stt[{index}]=skipped reason={stt.get('reason')}")
        else:
            terminal = stt.get("terminal_event") or {}
            data = terminal.get("data") or {}
            text = str(data.get("text") or "")
            print(
                f"stt[{index}]=frames={data.get('frames')} "
                f"duration_ms={data.get('audio_duration_ms')} "
                f"text_chars={len(text)}"
            )
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        human_summary(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
