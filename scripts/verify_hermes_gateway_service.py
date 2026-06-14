#!/usr/bin/env python3
"""Verify the running Hermes gateway is wired to the local voice stream path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_SERVICE_NAME = "hermes-gateway.service"
DEFAULT_SIDECAR_URL = "http://127.0.0.1:8787"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_voice_bin() -> str:
    env_value = os.environ.get("VOICE_BIN")
    if env_value:
        return env_value
    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return str(release_bin)
    return "voice"


def default_hermes_home() -> str:
    env_value = os.environ.get("HERMES_HOME")
    if env_value:
        return env_value
    return str(Path.home() / ".hermes")


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        return os.path.abspath(os.path.expanduser(value))

    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return os.path.abspath(found)


def resolve_command_executable(value: str) -> str | None:
    if "/" in value:
        return os.path.abspath(os.path.expanduser(value))
    found = shutil.which(value)
    return os.path.abspath(found) if found else None


def run_command(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def parse_systemctl_show(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def get_service_state(service_name: str, *, timeout: float) -> dict[str, str]:
    completed = run_command(
        [
            "systemctl",
            "--user",
            "show",
            service_name,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "Environment",
            "-p",
            "ExecStart",
            "-p",
            "WorkingDirectory",
            "-p",
            "FragmentPath",
            "-p",
            "DropInPaths",
            "--no-pager",
        ],
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"systemctl show failed for {service_name}: {detail}")
    return parse_systemctl_show(completed.stdout)


def parse_systemd_environment(value: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for part in shlex.split(value):
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        env[key] = raw
    return env


def parse_exec_start_argv(exec_start: str) -> list[str]:
    marker = "argv[]="
    if marker in exec_start:
        raw = exec_start.split(marker, 1)[1]
        raw = raw.split(" ;", 1)[0].split(" }", 1)[0]
        return shlex.split(raw)
    return shlex.split(exec_start)


def option_value(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name and index + 1 < len(args):
            return args[index + 1]
    return None


def validate_stream_command(command: str, *, voice_bin: str) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return [f"stream command is not shell-parseable: {exc}"], {}

    summary: dict[str, Any] = {"argv": args}
    if not args:
        return ["stream command must not be empty"], summary

    command_voice = resolve_command_executable(args[0])
    if command_voice != voice_bin:
        failures.append(
            f"stream command voice binary={command_voice!r}, expected {voice_bin!r}"
        )

    if len(args) < 2 or args[1] != "stream":
        failures.append("stream command must invoke `voice stream`")
    if "--quiet" not in args:
        failures.append("stream command must pass --quiet")

    expected_options = {
        "--sample-rate": "{sample_rate}",
        "--frame-ms": "{frame_ms}",
        "--raw-output": "-",
        "--input-file": "{input_path}",
    }
    for option, expected in expected_options.items():
        actual = option_value(args, option)
        if actual != expected:
            failures.append(
                f"stream command must pass {option} {expected}; got {actual or '<missing>'}"
            )

    for option in ("--voice", "--speed"):
        if option_value(args, option) is None:
            failures.append(f"stream command must pass {option}")

    return failures, summary


def validate_service(
    state: dict[str, str],
    *,
    service_name: str,
    hermes_home: Path,
    sidecar_url: str,
    voice_bin: str,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []

    active_state = state.get("ActiveState")
    if active_state != "active":
        failures.append(f"{service_name} ActiveState={active_state!r}, expected 'active'")

    try:
        main_pid = int(state.get("MainPID") or "0")
    except ValueError:
        main_pid = 0
    if main_pid <= 0:
        failures.append(f"{service_name} MainPID must be positive")

    exec_argv = parse_exec_start_argv(state.get("ExecStart", ""))
    if not exec_argv:
        failures.append(f"{service_name} ExecStart is empty")
    elif not (
        "hermes" in Path(exec_argv[0]).name
        or exec_argv[-3:] == ["hermes_cli.main", "gateway", "run"]
        or exec_argv[-4:] == ["-m", "hermes_cli.main", "gateway", "run"]
    ):
        failures.append(f"{service_name} ExecStart must run Hermes gateway")

    working_directory = state.get("WorkingDirectory", "")
    if working_directory and Path(working_directory).expanduser() != hermes_home:
        failures.append(
            f"{service_name} WorkingDirectory={working_directory!r}, "
            f"expected {str(hermes_home)!r}"
        )

    env = parse_systemd_environment(state.get("Environment", ""))
    if env.get("HERMES_HOME") != str(hermes_home):
        failures.append(
            f"{service_name} HERMES_HOME={env.get('HERMES_HOME')!r}, "
            f"expected {str(hermes_home)!r}"
        )
    if env.get("WHATSAPP_CLOUD_CALLING_SIDECAR_URL") != sidecar_url:
        failures.append(
            f"{service_name} WHATSAPP_CLOUD_CALLING_SIDECAR_URL="
            f"{env.get('WHATSAPP_CLOUD_CALLING_SIDECAR_URL')!r}, expected {sidecar_url!r}"
        )

    pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [Path(part).expanduser() for part in pythonpath.split(":") if part]
    if not pythonpath_entries:
        failures.append(f"{service_name} PYTHONPATH must include the local Hermes code path")
    elif not any(path.exists() for path in pythonpath_entries):
        failures.append(f"{service_name} PYTHONPATH entries do not exist: {pythonpath!r}")

    stream_command = env.get("WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND")
    if not stream_command:
        stream_failures: list[str] = [
            f"{service_name} WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND is missing"
        ]
        stream_summary: dict[str, Any] = {}
    else:
        stream_failures, stream_summary = validate_stream_command(
            stream_command,
            voice_bin=voice_bin,
        )
    failures.extend(f"{service_name}: {failure}" for failure in stream_failures)

    timeout = env.get("WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT")
    if timeout is not None:
        try:
            if float(timeout) <= 0:
                failures.append(
                    f"{service_name} WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT "
                    "must be positive"
                )
        except ValueError:
            failures.append(
                f"{service_name} WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT "
                f"is not numeric: {timeout!r}"
            )

    summary = {
        "active_state": active_state,
        "sub_state": state.get("SubState"),
        "main_pid": main_pid,
        "exec_start": exec_argv,
        "working_directory": working_directory,
        "hermes_home": env.get("HERMES_HOME"),
        "pythonpath": pythonpath,
        "sidecar_url": env.get("WHATSAPP_CLOUD_CALLING_SIDECAR_URL"),
        "stream_command": stream_summary,
        "fragment_path": state.get("FragmentPath"),
        "drop_in_paths": state.get("DropInPaths"),
    }
    return failures, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=default_voice_bin())
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--hermes-home", default=default_hermes_home())
    parser.add_argument("--sidecar-url", default=DEFAULT_SIDECAR_URL)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    hermes_home = Path(args.hermes_home).expanduser().resolve()
    sidecar_url = args.sidecar_url.rstrip("/")

    failures: list[str] = []
    checks: dict[str, Any] = {
        "service": args.service_name,
        "voice_bin": voice_bin,
        "hermes_home": str(hermes_home),
        "sidecar_url": sidecar_url,
    }
    try:
        state = get_service_state(args.service_name, timeout=args.timeout)
        service_failures, service_summary = validate_service(
            state,
            service_name=args.service_name,
            hermes_home=hermes_home,
            sidecar_url=sidecar_url,
            voice_bin=voice_bin,
        )
    except Exception as exc:
        service_failures = [str(exc)]
        service_summary = {}

    failures.extend(service_failures)
    checks["gateway_service"] = {
        **service_summary,
        "failures": service_failures,
    }

    result = {"success": not failures, "checks": checks, "failures": failures}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif failures:
        print("error: Hermes gateway service verifier failed", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
    else:
        print("ok: Hermes gateway service verifier passed")
        print(f"service={args.service_name}:checked")
        print(f"voice_bin={voice_bin}")
        print(f"hermes_home={hermes_home}")
        print(f"sidecar_url={sidecar_url}")
        print("stream_command=voice stream --raw-output -")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
