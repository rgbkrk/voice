#!/usr/bin/env python3
"""Summarize local voice-owned WhatsApp speech runtime readiness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_BRIDGE_URL = "http://127.0.0.1:3000"
DEFAULT_SIDECAR_URL = "http://127.0.0.1:8787"

META_SETUP_STEPS = (
    "Create or select a WhatsApp Business Platform app and WABA.",
    "Attach a phone number that is eligible for WhatsApp Cloud API.",
    "Generate a permanent System User access token with WhatsApp permissions.",
    "Configure webhook verify token and app secret for signed inbound webhooks.",
    "Enable/approve WhatsApp Calling for the Cloud phone number.",
    "Route Cloud Calling webhooks to the Hermes WhatsApp Cloud adapter.",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_voice_bin() -> str:
    env_value = os.environ.get("VOICE_BIN")
    if env_value:
        return env_value
    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return str(release_bin)
    found = shutil.which("voice")
    return found or "voice"


def default_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


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


def parse_json_stdout(completed: subprocess.CompletedProcess[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def output_failures(
    completed: subprocess.CompletedProcess[str],
    payload: dict[str, Any] | None,
) -> list[str]:
    if payload and isinstance(payload.get("failures"), list):
        return [str(item) for item in payload["failures"]]
    lines = [
        line.strip()
        for line in (completed.stderr or completed.stdout).splitlines()
        if line.strip()
    ]
    return lines[:8]


def component(
    *,
    name: str,
    category: str,
    command: list[str],
    timeout: float,
    required: bool = True,
    parse_json: bool = False,
) -> dict[str, Any]:
    try:
        completed = run_command(command, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "category": category,
            "required": required,
            "success": False,
            "returncode": None,
            "command": command,
            "failures": [f"timed out after {exc.timeout}s"],
            "summary": None,
        }

    payload = parse_json_stdout(completed) if parse_json else None
    if parse_json and payload is not None and isinstance(payload.get("success"), bool):
        success = bool(payload["success"])
    else:
        success = completed.returncode == 0
    return {
        "name": name,
        "category": category,
        "required": required,
        "success": success,
        "returncode": completed.returncode,
        "command": command,
        "failures": [] if success else output_failures(completed, payload),
        "summary": payload,
    }


def script_path(name: str) -> str:
    base = Path(os.environ.get("VOICE_READINESS_SCRIPT_DIR") or repo_root() / "scripts")
    path = base / name
    if not path.is_file():
        raise SystemExit(f"required script missing: {path}")
    return str(path)


def build_components(args: argparse.Namespace) -> list[dict[str, Any]]:
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")
    hermes_home = args.hermes_home.expanduser().resolve()
    hermes_config = args.hermes_config.expanduser().resolve()
    components: list[dict[str, Any]] = []

    hermes_config_cmd = [
        script_path("verify_hermes_voice_config.py"),
        "--config",
        str(hermes_config),
        "--voice-bin",
        voice_bin,
        "--text",
        args.text,
    ]
    if args.skip_hermes_tts_smoke:
        hermes_config_cmd.append("--skip-tts-smoke")
    components.append(
        component(
            name="hermes_voice_config",
            category="hermes_config",
            command=hermes_config_cmd,
            timeout=args.timeout,
        )
    )

    if not args.skip_systemd:
        components.append(
            component(
                name="hermes_gateway_service",
                category="hermes_runtime",
                command=[
                    script_path("verify_hermes_gateway_service.py"),
                    "--voice-bin",
                    voice_bin,
                    "--hermes-home",
                    str(hermes_home),
                    "--sidecar-url",
                    args.sidecar_url,
                    "--json",
                ],
                timeout=args.timeout,
                parse_json=True,
            )
        )

    cli_cmd = [script_path("verify_cli_mcp_surface.py"), "--voice-bin", voice_bin]
    if args.skip_daemon:
        cli_cmd.append("--skip-daemon")
    else:
        cli_cmd.append("--require-daemon")
    components.append(
        component(
            name="voice_cli_mcp",
            category="voice_runtime",
            command=cli_cmd,
            timeout=args.timeout,
            parse_json=True,
        )
    )

    whatsapp_contract_cmd = [
        script_path("verify_whatsapp_voice_contract.sh"),
        "--voice-bin",
        voice_bin,
        "--text",
        args.text,
    ]
    if args.skip_daemon:
        whatsapp_contract_cmd.append("--skip-daemon")
    else:
        whatsapp_contract_cmd.append("--require-daemon")
        if args.run_stt_smoke:
            whatsapp_contract_cmd.append("--run-stt-smoke")
    components.append(
        component(
            name="voice_whatsapp_contract",
            category="voice_runtime",
            command=whatsapp_contract_cmd,
            timeout=args.voice_timeout,
        )
    )

    bridge_cmd = [
        script_path("verify_whatsapp_bridge_runtime.py"),
        "--hermes-home",
        str(hermes_home),
        "--bridge-url",
        args.bridge_url,
        "--json",
    ]
    if args.expected_agent_number:
        bridge_cmd.extend(["--expected-agent-number", args.expected_agent_number])
    if args.expected_agent_name:
        bridge_cmd.extend(["--expected-agent-name", args.expected_agent_name])
    if args.skip_systemd:
        bridge_cmd.append("--skip-systemd")
    components.append(
        component(
            name="whatsapp_bridge_runtime",
            category="bridge_pairing",
            command=bridge_cmd,
            timeout=args.timeout,
            parse_json=True,
        )
    )

    if not args.skip_voice_note_smoke:
        components.append(
            component(
                name="whatsapp_voice_note_dry_run",
                category="voice_note",
                command=[
                    script_path("verify_whatsapp_voice_note_bridge.py"),
                    "--voice-bin",
                    voice_bin,
                    "--hermes-home",
                    str(hermes_home),
                    "--bridge-url",
                    args.bridge_url,
                    "--text",
                    args.text,
                    "--json",
                ],
                timeout=args.voice_timeout,
                parse_json=True,
            )
        )

    if args.run_inbound_cache_smoke:
        inbound_cmd = [
            script_path("verify_whatsapp_inbound_audio_cache.py"),
            "--voice-bin",
            voice_bin,
            "--hermes-home",
            str(hermes_home),
            "--require-cache",
            "--run-stt",
            "--json",
        ]
        if args.whatsapp_audio_cache_dir:
            inbound_cmd.extend(["--audio-cache-dir", str(args.whatsapp_audio_cache_dir)])
        components.append(
            component(
                name="whatsapp_inbound_cache_stt",
                category="voice_note",
                command=inbound_cmd,
                timeout=args.stt_timeout,
                parse_json=True,
            )
        )

    if not args.skip_sidecar:
        sidecar_cmd = [
            script_path("verify_webrtc_sidecar_service.py"),
            "--voice-bin",
            voice_bin,
            "--sidecar-url",
            args.sidecar_url,
            "--json",
        ]
        if args.skip_systemd:
            sidecar_cmd.append("--skip-systemd")
        components.append(
            component(
                name="webrtc_sidecar",
                category="live_call_local",
                command=sidecar_cmd,
                timeout=args.timeout,
                parse_json=True,
            )
        )

    return components


def bridge_cloud_summary(components: list[dict[str, Any]]) -> dict[str, Any]:
    for item in components:
        if item["name"] != "whatsapp_bridge_runtime":
            continue
        summary = item.get("summary") or {}
        checks = summary.get("checks") or {}
        cloud = checks.get("whatsapp_cloud")
        return cloud if isinstance(cloud, dict) else {}
    return {}


def build_readiness(args: argparse.Namespace) -> dict[str, Any]:
    components = build_components(args)
    required_failures = [
        item for item in components if item.get("required") and not item.get("success")
    ]
    by_category: dict[str, dict[str, Any]] = {}
    for item in components:
        category = item["category"]
        bucket = by_category.setdefault(category, {"success": True, "components": []})
        bucket["components"].append(item["name"])
        if item.get("required") and not item.get("success"):
            bucket["success"] = False

    cloud = bridge_cloud_summary(components)
    cloud_missing = [str(key) for key in cloud.get("cloud_missing") or []]
    calling_missing = [str(key) for key in cloud.get("calling_missing") or []]
    external_meta_setup = {
        "cloud_configured": bool(cloud.get("cloud_configured")),
        "calling_sidecar_configured": bool(cloud.get("calling_sidecar_configured")),
        "calling_ready": bool(cloud.get("calling_ready")),
        "cloud_missing": cloud_missing,
        "calling_missing": calling_missing,
        "setup_steps": list(META_SETUP_STEPS) if cloud_missing or calling_missing else [],
    }

    external_failures: list[dict[str, Any]] = []
    success = not required_failures
    if args.require_whatsapp_calling and not external_meta_setup["calling_ready"]:
        success = False
        external_failures.append(
            {
                "name": "whatsapp_cloud_calling",
                "category": "external_meta_setup",
                "failures": [
                    "WhatsApp Cloud Calling not ready; missing: "
                    + (", ".join(calling_missing) or "external Meta calling approval")
                ],
            }
        )
    if args.require_whatsapp_cloud and not external_meta_setup["cloud_configured"]:
        success = False
        external_failures.append(
            {
                "name": "whatsapp_cloud",
                "category": "external_meta_setup",
                "failures": [
                    "WhatsApp Cloud credentials missing: "
                    + (", ".join(cloud_missing) or "external Meta Cloud setup")
                ],
            }
        )

    return {
        "success": success,
        "components": components,
        "by_category": by_category,
        "external_meta_setup": external_meta_setup,
        "failures": [
            {
                "name": item["name"],
                "category": item["category"],
                "failures": item.get("failures") or [],
            }
            for item in required_failures
        ]
        + external_failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-bin", default=default_voice_bin())
    parser.add_argument("--hermes-home", type=Path, default=default_hermes_home())
    parser.add_argument(
        "--hermes-config",
        type=Path,
        default=default_hermes_home() / "config.yaml",
    )
    parser.add_argument("--bridge-url", default=os.environ.get("WHATSAPP_BRIDGE_URL", DEFAULT_BRIDGE_URL))
    parser.add_argument("--sidecar-url", default=DEFAULT_SIDECAR_URL)
    parser.add_argument("--whatsapp-audio-cache-dir", type=Path, default=None)
    parser.add_argument("--expected-agent-number", default=os.environ.get("WHATSAPP_AGENT_NUMBER"))
    parser.add_argument("--expected-agent-name", default=os.environ.get("WHATSAPP_AGENT_NAME"))
    parser.add_argument("--text", default="WhatsApp alpha readiness smoke.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--voice-timeout", type=float, default=180.0)
    parser.add_argument("--stt-timeout", type=float, default=180.0)
    parser.add_argument("--skip-systemd", action="store_true")
    parser.add_argument("--skip-daemon", action="store_true")
    parser.add_argument("--skip-sidecar", action="store_true")
    parser.add_argument("--skip-hermes-tts-smoke", action="store_true")
    parser.add_argument("--skip-voice-note-smoke", action="store_true")
    parser.add_argument("--run-stt-smoke", action="store_true")
    parser.add_argument("--run-inbound-cache-smoke", action="store_true")
    parser.add_argument("--require-whatsapp-cloud", action="store_true")
    parser.add_argument("--require-whatsapp-calling", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def human_summary(result: dict[str, Any]) -> None:
    if result["success"]:
        print("ok: WhatsApp alpha readiness passed")
    else:
        print("error: WhatsApp alpha readiness failed", file=sys.stderr)

    for item in result["components"]:
        status = "ok" if item["success"] else "failed"
        print(f"{item['name']}={status} category={item['category']}")
        if not item["success"]:
            for failure in item.get("failures") or []:
                print(f"  - {failure}", file=sys.stderr)

    external = result["external_meta_setup"]
    print(
        "whatsapp_cloud="
        + ("configured" if external["cloud_configured"] else "not_configured")
        + " missing="
        + (",".join(external["cloud_missing"]) or "none")
    )
    print(
        "whatsapp_calling="
        + ("ready" if external["calling_ready"] else "not_ready")
        + " missing="
        + (",".join(external["calling_missing"]) or "none")
    )
    if external["setup_steps"]:
        print("external_meta_setup=required")
        for step in external["setup_steps"]:
            print(f"- {step}")
    else:
        print("external_meta_setup=complete")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_readiness(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        human_summary(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
