#!/usr/bin/env python3
"""Verify a local voice WebRTC sidecar service.

This is the companion verifier for ``install_webrtc_sidecar_service.sh``. It
checks that the sidecar HTTP contract matches the installed ``voice
stream-contract`` output, that the sidecar health endpoint reports the same
audio shape, and, on Linux/systemd hosts, that the user services are active.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Any
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen


DEFAULT_SIDECAR_URL = "http://127.0.0.1:8787"
DEFAULT_SERVICE_NAME = "voice-webrtc-sidecar.service"
DEFAULT_DAEMON_SERVICE = "voiced.service"
EXPECTED_AUDIO = {
    "sample_rate": 48_000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le",
    "bytes_per_sample": 2,
    "samples_per_frame": 960,
    "frame_bytes": 1_920,
}
REQUIRED_VOICE_SURFACES = (
    "completed_voice_note",
    "streamed_voice_note",
    "raw_outbound_pcm",
    "raw_inbound_pcm",
    "file_transcription_smoke",
)
REQUIRED_ENDPOINTS = {
    "contract": ("GET", "/contract"),
    "health": ("GET", "/health"),
    "offer": ("POST", "/offer"),
    "call_status": ("GET", "/calls/{call_id}"),
    "receive_audio": ("GET", "/calls/{call_id}/audio"),
    "send_audio": ("POST", "/calls/{call_id}/audio"),
    "clear_audio": ("POST", "/calls/{call_id}/audio/clear"),
    "close_call": ("POST", "/calls/{call_id}/close"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_executable(value: str, *, label: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} is not executable: {path}")
        return os.path.abspath(os.path.expanduser(value))

    found = shutil.which(value)
    if not found:
        raise SystemExit(f"{label} not found on PATH: {value}")
    return found


def default_voice_bin() -> str:
    env_value = os.environ.get("VOICE_BIN")
    if env_value:
        return env_value
    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return str(release_bin)
    return "voice"


def run_command(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def load_voice_contract(voice_bin: str, *, timeout: float) -> dict[str, Any]:
    completed = run_command([voice_bin, "stream-contract"], timeout=timeout)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"`voice stream-contract` failed: {detail}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("`voice stream-contract` did not return JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("`voice stream-contract` JSON root must be an object")
    return parsed


def fetch_json(base_url: str, path: str, *, timeout: float) -> dict[str, Any]:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    try:
        with urlopen(url, timeout=timeout) as response:
            raw = response.read()
    except URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GET {url} did not return JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"GET {url} JSON root must be an object")
    return parsed


def validate_contract_shape(contract: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if contract.get("contract") != "voice.webrtc_sidecar":
        failures.append("contract id must be voice.webrtc_sidecar")
    if contract.get("version") != 1:
        failures.append("contract version must be 1")

    audio = contract.get("audio")
    if not isinstance(audio, dict):
        failures.append("contract audio section must be an object")
    else:
        for key, expected in EXPECTED_AUDIO.items():
            if audio.get(key) != expected:
                failures.append(f"audio.{key}={audio.get(key)!r}, expected {expected!r}")
        default_drain = int(audio.get("default_drain_bytes") or 0)
        max_tx = int(audio.get("max_outbound_queue_bytes") or 0)
        max_rx = int(audio.get("max_inbound_queue_bytes") or 0)
        frame_bytes = int(audio.get("frame_bytes") or 0)
        if frame_bytes > 0 and default_drain % frame_bytes != 0:
            failures.append("audio.default_drain_bytes must align to whole frames")
        if max_tx < frame_bytes:
            failures.append("audio.max_outbound_queue_bytes must fit at least one frame")
        if max_rx < default_drain:
            failures.append("audio.max_inbound_queue_bytes must fit the default drain")

    surfaces = contract.get("voice_surfaces")
    if not isinstance(surfaces, dict):
        failures.append("contract voice_surfaces section must be an object")
    else:
        for key in REQUIRED_VOICE_SURFACES:
            if key not in surfaces:
                failures.append(f"missing voice_surfaces.{key}")
        completed = surfaces.get("completed_voice_note") or {}
        if completed.get("output") != "audio/ogg; codecs=opus":
            failures.append("completed_voice_note output must be audio/ogg; codecs=opus")
        raw_outbound = surfaces.get("raw_outbound_pcm") or {}
        if raw_outbound.get("frame_bytes") != EXPECTED_AUDIO["frame_bytes"]:
            failures.append("raw_outbound_pcm frame_bytes must be 1920")

    endpoints = contract.get("endpoints")
    if not isinstance(endpoints, dict):
        failures.append("contract endpoints section must be an object")
    else:
        for key, (method, path) in REQUIRED_ENDPOINTS.items():
            endpoint = endpoints.get(key)
            if not isinstance(endpoint, dict):
                failures.append(f"missing endpoints.{key}")
                continue
            if endpoint.get("method") != method:
                failures.append(f"endpoints.{key}.method must be {method}")
            if endpoint.get("path") != path:
                failures.append(f"endpoints.{key}.path must be {path}")

    return failures


def validate_health(health: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if health.get("ok") is not True:
        failures.append("sidecar /health ok must be true")
    if not isinstance(health.get("sessions"), int):
        failures.append("sidecar /health sessions must be an integer")
    if not isinstance(health.get("call_ids"), list):
        failures.append("sidecar /health call_ids must be a list")
    if health.get("audio") != contract.get("audio"):
        failures.append("sidecar /health audio does not match /contract audio")
    return failures


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
            "After",
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


def option_values(argv: list[str], option: str) -> list[str]:
    values: list[str] = []
    prefix = f"{option}="
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif arg.startswith(prefix):
            values.append(arg.removeprefix(prefix))
    return values


def validate_sidecar_service(
    service_state: dict[str, str],
    *,
    service_name: str,
    sidecar_url: str,
    voice_bin: str,
    expected_repo_root: Path | None,
    expected_daemon_service: str | None,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    parsed_url = urlparse(sidecar_url)
    expected_host = parsed_url.hostname or ""
    expected_port = str(parsed_url.port or (443 if parsed_url.scheme == "https" else 80))

    active_state = service_state.get("ActiveState")
    if active_state != "active":
        failures.append(f"{service_name} ActiveState={active_state!r}, expected 'active'")

    try:
        main_pid = int(service_state.get("MainPID") or "0")
    except ValueError:
        main_pid = 0
    if main_pid <= 0:
        failures.append(f"{service_name} MainPID must be positive")

    env = parse_systemd_environment(service_state.get("Environment", ""))
    if env.get("VOICE_BIN") != voice_bin:
        failures.append(
            f"{service_name} VOICE_BIN={env.get('VOICE_BIN')!r}, expected {voice_bin!r}"
        )

    argv = parse_exec_start_argv(service_state.get("ExecStart", ""))
    if not argv:
        failures.append(f"{service_name} ExecStart is empty")
    else:
        if not any(arg.endswith("sidecar.py") for arg in argv):
            failures.append(f"{service_name} ExecStart must run sidecar.py")
        if expected_host and expected_host not in option_values(argv, "--host"):
            failures.append(f"{service_name} ExecStart must include --host {expected_host}")
        if expected_port not in option_values(argv, "--port"):
            failures.append(f"{service_name} ExecStart must include --port {expected_port}")

    working_directory = service_state.get("WorkingDirectory", "")
    if expected_repo_root is not None and working_directory != str(expected_repo_root):
        failures.append(
            f"{service_name} WorkingDirectory={working_directory!r}, "
            f"expected {str(expected_repo_root)!r}"
        )

    after_units = set(service_state.get("After", "").split())
    if expected_daemon_service and expected_daemon_service not in after_units:
        failures.append(f"{service_name} After= must include {expected_daemon_service}")

    summary = {
        "active_state": active_state,
        "sub_state": service_state.get("SubState"),
        "main_pid": main_pid,
        "voice_bin": env.get("VOICE_BIN"),
        "exec_start": argv,
        "working_directory": working_directory,
    }
    return failures, summary


def validate_active_service(
    service_state: dict[str, str],
    *,
    service_name: str,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    active_state = service_state.get("ActiveState")
    if active_state != "active":
        failures.append(f"{service_name} ActiveState={active_state!r}, expected 'active'")
    try:
        main_pid = int(service_state.get("MainPID") or "0")
    except ValueError:
        main_pid = 0
    if main_pid <= 0:
        failures.append(f"{service_name} MainPID must be positive")
    return failures, {
        "active_state": active_state,
        "sub_state": service_state.get("SubState"),
        "main_pid": main_pid,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=default_voice_bin())
    parser.add_argument("--sidecar-url", default=DEFAULT_SIDECAR_URL)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--voice-daemon-service", default=DEFAULT_DAEMON_SERVICE)
    parser.add_argument(
        "--repo-root",
        default=str(repo_root()),
        help="expected sidecar WorkingDirectory (default: current voice checkout)",
    )
    parser.add_argument(
        "--skip-systemd",
        action="store_true",
        help="skip systemd user-service checks and only verify HTTP/voice contracts",
    )
    parser.add_argument(
        "--skip-daemon-service",
        action="store_true",
        help="skip the voice daemon service active check",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    sidecar_url = args.sidecar_url.rstrip("/")
    expected_repo_root = None if args.skip_systemd else Path(args.repo_root).resolve()

    failures: list[str] = []
    checks: dict[str, Any] = {
        "voice_bin": voice_bin,
        "sidecar_url": sidecar_url,
    }

    try:
        voice_contract = load_voice_contract(voice_bin, timeout=args.timeout)
    except Exception as exc:
        failures.append(str(exc))
        voice_contract = {}
    voice_contract_failures = validate_contract_shape(voice_contract) if voice_contract else []
    failures.extend(f"voice contract: {failure}" for failure in voice_contract_failures)
    checks["voice_contract"] = {
        "loaded": bool(voice_contract),
        "failures": voice_contract_failures,
    }

    try:
        sidecar_contract = fetch_json(sidecar_url, "/contract", timeout=args.timeout)
    except Exception as exc:
        failures.append(str(exc))
        sidecar_contract = {}
    sidecar_contract_failures = (
        validate_contract_shape(sidecar_contract) if sidecar_contract else []
    )
    failures.extend(f"sidecar contract: {failure}" for failure in sidecar_contract_failures)
    contract_matches = bool(voice_contract) and voice_contract == sidecar_contract
    if voice_contract and sidecar_contract and not contract_matches:
        failures.append("sidecar /contract does not match `voice stream-contract`")
    checks["sidecar_contract"] = {
        "loaded": bool(sidecar_contract),
        "matches_voice": contract_matches,
        "failures": sidecar_contract_failures,
    }

    try:
        health = fetch_json(sidecar_url, "/health", timeout=args.timeout)
    except Exception as exc:
        failures.append(str(exc))
        health = {}
    health_failures = validate_health(health, sidecar_contract) if health else []
    failures.extend(f"sidecar health: {failure}" for failure in health_failures)
    checks["sidecar_health"] = {
        "loaded": bool(health),
        "ok": health.get("ok") if health else None,
        "sessions": health.get("sessions") if health else None,
        "audio": health.get("audio") if health else None,
        "failures": health_failures,
    }

    if not args.skip_systemd:
        try:
            state = get_service_state(args.service_name, timeout=args.timeout)
            service_failures, service_summary = validate_sidecar_service(
                state,
                service_name=args.service_name,
                sidecar_url=sidecar_url,
                voice_bin=voice_bin,
                expected_repo_root=expected_repo_root,
                expected_daemon_service=(
                    None if args.skip_daemon_service else args.voice_daemon_service
                ),
            )
        except Exception as exc:
            service_failures = [str(exc)]
            service_summary = {}
        failures.extend(f"{args.service_name}: {failure}" for failure in service_failures)
        checks["sidecar_service"] = {
            "service": args.service_name,
            **service_summary,
            "failures": service_failures,
        }

        if not args.skip_daemon_service:
            try:
                state = get_service_state(args.voice_daemon_service, timeout=args.timeout)
                daemon_failures, daemon_summary = validate_active_service(
                    state,
                    service_name=args.voice_daemon_service,
                )
            except Exception as exc:
                daemon_failures = [str(exc)]
                daemon_summary = {}
            failures.extend(
                f"{args.voice_daemon_service}: {failure}" for failure in daemon_failures
            )
            checks["voice_daemon_service"] = {
                "service": args.voice_daemon_service,
                **daemon_summary,
                "failures": daemon_failures,
            }
    else:
        checks["sidecar_service"] = {"skipped": True}
        checks["voice_daemon_service"] = {"skipped": True}

    result = {
        "success": not failures,
        "checks": checks,
        "failures": failures,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif failures:
        print("error: voice WebRTC sidecar service verifier failed", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
    else:
        print("ok: voice WebRTC sidecar service verifier passed")
        print(f"voice_bin={voice_bin}")
        print(f"sidecar_url={sidecar_url}")
        print("contract=matched")
        if args.skip_systemd:
            print("systemd=skipped")
        else:
            print(f"sidecar_service={args.service_name}:checked")
            if args.skip_daemon_service:
                print("voice_daemon_service=skipped")
            else:
                print(f"voice_daemon_service={args.voice_daemon_service}:checked")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
