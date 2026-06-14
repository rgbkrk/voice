#!/usr/bin/env python3
"""Start a long non-draining WhatsApp attended receive cache watch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_WAIT_SECONDS = 6 * 60 * 60
DEFAULT_UNIT_PREFIX = "voice-whatsapp-attended-cache-watch"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def default_hermes_config() -> Path:
    env_value = os.environ.get("HERMES_CONFIG")
    if env_value:
        return Path(env_value)
    return default_hermes_home() / "config.yaml"


def default_voice_bin() -> str:
    env_value = os.environ.get("VOICE_BIN")
    if env_value:
        return env_value
    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return str(release_bin)
    found = shutil.which("voice")
    return found or "voice"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_alpha_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.alpha_script),
        "--voice-bin",
        str(args.voice_bin),
        "--hermes-home",
        str(args.hermes_home),
        "--hermes-config",
        str(args.hermes_config),
        "--profile",
        "attended-cache-receive",
        "--wait-audio-cache-seconds",
        str(float(args.wait_seconds)),
    ]
    if args.expected_agent_number:
        command.extend(["--expected-agent-number", args.expected_agent_number])
    if args.expected_agent_name:
        command.extend(["--expected-agent-name", args.expected_agent_name])
    command.append("--json")
    return command


def build_watch(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = args.timestamp or utc_timestamp()
    unit = f"{args.unit_prefix}-{timestamp}"
    service = unit if unit.endswith(".service") else f"{unit}.service"
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / f"{unit}.json"
    log_path = output_dir / f"{unit}.log"
    alpha_command = build_alpha_command(args)
    inner_command = (
        f"cd {shlex.quote(str(args.repo_root))} && "
        f"{shlex.join(alpha_command)} > {shlex.quote(str(json_path))} "
        f"2> {shlex.quote(str(log_path))}"
    )
    systemd_command = [
        str(args.systemd_run_bin),
        "--user",
        f"--unit={unit}",
        "--collect",
        "/bin/bash",
        "-lc",
        inner_command,
    ]
    return {
        "unit": unit,
        "service": service,
        "json_path": str(json_path),
        "log_path": str(log_path),
        "alpha_command": alpha_command,
        "systemd_command": systemd_command,
        "status_command": ["systemctl", "--user", "status", service],
        "journal_command": ["journalctl", "--user", "-u", service, "-f"],
    }


def validate_args(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.wait_seconds <= 0:
        failures.append("--wait-seconds must be positive")
    if not args.unit_prefix:
        failures.append("--unit-prefix must not be empty")
    if "/" in args.unit_prefix:
        failures.append("--unit-prefix must not contain '/'")
    return failures


def print_human(result: dict[str, Any]) -> None:
    if result.get("dry_run"):
        print("ok: WhatsApp attended cache watch dry run")
    else:
        print("ok: WhatsApp attended cache watch started")
    print(f"unit={result['unit']}")
    print(f"service={result['service']}")
    print(f"json={result['json_path']}")
    print(f"log={result['log_path']}")
    print(f"wait_seconds={result['wait_seconds']}")
    print(f"alpha_command={shlex.join(result['alpha_command'])}")
    print(f"systemd_command={shlex.join(result['systemd_command'])}")
    print(f"status_command={shlex.join(result['status_command'])}")
    print(f"journal_command={shlex.join(result['journal_command'])}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=default_voice_bin())
    parser.add_argument("--hermes-home", type=Path, default=default_hermes_home())
    parser.add_argument("--hermes-config", type=Path, default=default_hermes_config())
    parser.add_argument("--wait-seconds", type=float, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--expected-agent-number", default=os.environ.get("WHATSAPP_AGENT_NUMBER"))
    parser.add_argument("--expected-agent-name", default=os.environ.get("WHATSAPP_AGENT_NAME"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument(
        "--alpha-script",
        type=Path,
        default=repo_root() / "scripts" / "verify_whatsapp_alpha_readiness.py",
    )
    parser.add_argument("--systemd-run-bin", default=os.environ.get("SYSTEMD_RUN_BIN", "systemd-run"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.hermes_home = args.hermes_home.expanduser().resolve()
    args.hermes_config = args.hermes_config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.alpha_script = args.alpha_script.expanduser().resolve()

    failures = validate_args(args)
    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 2

    watch = build_watch(args)
    result = {
        **watch,
        "dry_run": args.dry_run,
        "wait_seconds": args.wait_seconds,
        "returncode": 0,
    }
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            watch["systemd_command"],
            check=False,
            text=True,
            capture_output=True,
        )
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
        if completed.returncode != 0:
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print_human(result)
                if completed.stderr.strip():
                    print(completed.stderr.strip(), file=sys.stderr)
            return completed.returncode

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
