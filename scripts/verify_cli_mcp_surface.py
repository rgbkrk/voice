#!/usr/bin/env python3
"""Verify fast CLI/MCP surfaces with and without the voice daemon.

This is intentionally lighter than ``macos_release_compare.py``: it does not
run synthesis, transcription, or model loading. It checks the control surfaces
that should remain usable before and after daemon installation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


MCP_SMOKE_INPUT = (
    '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}\n'
    '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}\n'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify voice CLI/MCP daemon detection surfaces."
    )
    parser.add_argument(
        "--voice-bin",
        type=Path,
        default=None,
        help="voice binary to verify; defaults to target/release/voice or PATH",
    )
    parser.add_argument(
        "--require-daemon",
        action="store_true",
        help="fail when no daemon is detected for the MCP daemon smoke",
    )
    parser.add_argument(
        "--skip-daemon",
        action="store_true",
        help="skip daemon-detected MCP smoke even if a daemon is running",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_voice_bin(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return release_bin

    found = shutil.which("voice")
    if found:
        return Path(found).resolve()

    raise SystemExit("voice binary not found; pass --voice-bin")


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def hidden_daemon_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VOICE_DAEMON_SOCKET"] = f"/tmp/voice-cli-mcp-missing-{os.getpid()}.sock"
    return env


def mcp_initialized(stdout: str) -> bool:
    return '"serverInfo"' in stdout and '"tools"' in stdout


def mcp_connected_to_daemon(stderr: str) -> bool:
    return (
        "voice mcp: connected to voice daemon" in stderr
        or "voice mcp: reconnected to voice daemon" in stderr
    )


def verify_no_daemon_surfaces(voice_bin: Path, timeout: float) -> list[dict[str, Any]]:
    env = hidden_daemon_env()
    checks: list[dict[str, Any]] = []

    contract = run_command(
        [str(voice_bin), "stream-contract"],
        env=env,
        timeout=timeout,
    )
    contract_ok = False
    contract_name = None
    if contract.returncode == 0:
        try:
            contract_json = json.loads(contract.stdout)
            contract_name = contract_json.get("contract")
            contract_ok = contract_name == "voice.webrtc_sidecar"
        except json.JSONDecodeError:
            contract_ok = False
    checks.append(
        {
            "name": "stream_contract_no_daemon",
            "ok": contract_ok,
            "returncode": contract.returncode,
            "contract": contract_name,
        }
    )

    mcp = run_command(
        [str(voice_bin), "mcp", "-q"],
        env=env,
        input_text=MCP_SMOKE_INPUT,
        timeout=timeout,
    )
    checks.append(
        {
            "name": "mcp_no_daemon_initializes",
            "ok": mcp.returncode == 0 and mcp_initialized(mcp.stdout),
            "returncode": mcp.returncode,
        }
    )

    return checks


def verify_daemon_surfaces(
    voice_bin: Path,
    *,
    require_daemon: bool,
    skip_daemon: bool,
    timeout: float,
) -> list[dict[str, Any]]:
    if skip_daemon:
        return [
            {
                "name": "daemon_detected",
                "ok": True,
                "skipped": True,
                "note": "--skip-daemon was provided",
            },
            {
                "name": "mcp_with_daemon_detects_daemon",
                "ok": True,
                "skipped": True,
                "note": "--skip-daemon was provided",
            },
        ]

    daemon = run_command(
        [str(voice_bin), "daemon", "status", "--json"],
        timeout=timeout,
    )
    daemon_available = daemon.returncode == 0
    checks: list[dict[str, Any]] = [
        {
            "name": "daemon_detected",
            "ok": daemon_available or not require_daemon,
            "detected": daemon_available,
            "returncode": daemon.returncode,
            "note": "MCP daemon smoke runs only when a daemon is detected",
        }
    ]

    if not daemon_available:
        checks.append(
            {
                "name": "mcp_with_daemon_detects_daemon",
                "ok": not require_daemon,
                "skipped": True,
                "note": "daemon not detected",
            }
        )
        return checks

    mcp = run_command(
        [str(voice_bin), "mcp"],
        input_text=MCP_SMOKE_INPUT,
        timeout=timeout,
    )
    checks.append(
        {
            "name": "mcp_with_daemon_detects_daemon",
            "ok": (
                mcp.returncode == 0
                and mcp_initialized(mcp.stdout)
                and mcp_connected_to_daemon(mcp.stderr)
            ),
            "returncode": mcp.returncode,
        }
    )
    return checks


def verify(args: argparse.Namespace) -> dict[str, Any]:
    voice_bin = resolve_voice_bin(args.voice_bin)
    checks = verify_no_daemon_surfaces(voice_bin, timeout=args.timeout)
    checks.extend(
        verify_daemon_surfaces(
            voice_bin,
            require_daemon=args.require_daemon,
            skip_daemon=args.skip_daemon,
            timeout=args.timeout,
        )
    )
    return {
        "success": all(bool(check.get("ok")) for check in checks),
        "voice_bin": str(voice_bin),
        "checks": checks,
    }


def main() -> int:
    result = verify(parse_args())
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
