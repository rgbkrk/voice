#!/usr/bin/env python3
"""Summarize local voice-owned WhatsApp speech runtime readiness."""

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


DEFAULT_BRIDGE_URL = "http://127.0.0.1:3000"
DEFAULT_SIDECAR_URL = "http://127.0.0.1:8787"
PROFILE_CHOICES = (
    "unattended",
    "cached-receive",
    "send",
    "attended-cache-receive",
    "attended-send-receive",
)

META_SETUP_STEPS = (
    "Create or select a WhatsApp Business Platform app and WABA.",
    "Attach a phone number that is eligible for WhatsApp Cloud API.",
    "Generate a permanent System User access token with WhatsApp permissions.",
    "Configure webhook verify token and app secret for signed inbound webhooks.",
    "Enable/approve WhatsApp Calling for the Cloud phone number.",
    "Route Cloud Calling webhooks to the Hermes WhatsApp Cloud adapter.",
)
CLOUD_REQUIRED_KEYS = (
    "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
    "WHATSAPP_CLOUD_ACCESS_TOKEN",
    "WHATSAPP_CLOUD_APP_SECRET",
    "WHATSAPP_CLOUD_VERIFY_TOKEN",
)
CALLING_SIDECAR_REQUIRED_KEYS = (
    "WHATSAPP_CLOUD_CALLING_SIDECAR_URL",
    "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_voice_bin() -> str:
    env_value = os.environ.get("VOICE_BIN")
    if env_value:
        return env_value
    found = shutil.which("voice")
    if found:
        return found
    release_bin = repo_root() / "target" / "release" / "voice"
    if release_bin.is_file() and os.access(release_bin, os.X_OK):
        return str(release_bin)
    return "voice"


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
        voice_note_cmd = [
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
        ]
        voice_note_name = "whatsapp_voice_note_dry_run"
        if args.voice_note_chat_id:
            voice_note_cmd.extend(["--chat-id", args.voice_note_chat_id])
        if args.send_voice_note:
            voice_note_cmd.append("--send")
            voice_note_name = "whatsapp_voice_note_send"
        if args.wait_inbound_seconds > 0:
            voice_note_cmd.extend(
                ["--wait-inbound-seconds", str(args.wait_inbound_seconds)]
            )
            if args.send_voice_note:
                voice_note_name = "whatsapp_voice_note_send_receive"
            else:
                voice_note_name = "whatsapp_voice_note_receive"
            if args.require_inbound_audio:
                voice_note_cmd.append("--require-inbound-audio")
            if args.drain_bridge_messages:
                voice_note_cmd.append("--drain-bridge-messages")
        voice_note_timeout = args.voice_timeout + args.wait_inbound_seconds + args.timeout
        components.append(
            component(
                name=voice_note_name,
                category="voice_note",
                command=voice_note_cmd,
                timeout=voice_note_timeout,
                parse_json=True,
            )
        )

    if args.run_inbound_cache_smoke:
        inbound_name = "whatsapp_inbound_cache_stt"
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
        if args.wait_audio_cache_seconds > 0:
            inbound_name = "whatsapp_inbound_cache_fresh_stt"
            inbound_cmd.extend(["--wait-fresh-seconds", str(args.wait_audio_cache_seconds)])
            if args.require_fresh_cache_audio:
                inbound_cmd.append("--require-fresh-audio")
        components.append(
            component(
                name=inbound_name,
                category="voice_note",
                command=inbound_cmd,
                timeout=args.stt_timeout + args.wait_audio_cache_seconds + args.timeout,
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
    checks = bridge_runtime_checks(components)
    cloud = checks.get("whatsapp_cloud")
    return cloud if isinstance(cloud, dict) else {}


def bridge_runtime_checks(components: list[dict[str, Any]]) -> dict[str, Any]:
    for item in components:
        if item["name"] != "whatsapp_bridge_runtime":
            continue
        summary = item.get("summary") or {}
        checks = summary.get("checks") or {}
        return checks if isinstance(checks, dict) else {}
    return {}


def attended_receive_gate(
    components: list[dict[str, Any]],
    *,
    hermes_home: Path,
    audio_cache_dir: Path,
) -> dict[str, Any]:
    bridge_checks = bridge_runtime_checks(components)
    local_config = bridge_checks.get("whatsapp_local_config") or {}
    identity = bridge_checks.get("baileys_identity") or {}
    receive_names = {"whatsapp_voice_note_receive", "whatsapp_voice_note_send_receive"}
    cached_receive_verified = any(
        item["name"] == "whatsapp_inbound_cache_stt" and item.get("success")
        for item in components
    )

    gate: dict[str, Any] = {
        "status": "pending_attended",
        "component": None,
        "cached_receive_verified": cached_receive_verified,
        "requires_operator": True,
        "drains_bridge_messages": False,
        "reason": "fresh WhatsApp inbound voice-note receive has not been run",
        "command": [
            "scripts/verify_whatsapp_alpha_readiness.py",
            "--hermes-home",
            str(hermes_home),
            "--profile",
            "attended-cache-receive",
        ],
        "fallback_draining_command": [
            "scripts/verify_whatsapp_alpha_readiness.py",
            "--hermes-home",
            str(hermes_home),
            "--profile",
            "attended-send-receive",
        ],
    }
    gate["operator_handoff"] = {
        "preferred_profile": "attended-cache-receive",
        "preferred_command": gate["command"],
        "fallback_profile": "attended-send-receive",
        "fallback_draining_command": gate["fallback_draining_command"],
        "drains_bridge_messages": False,
        "fallback_drains_bridge_messages": True,
        "audio_cache_dir": str(audio_cache_dir),
        "home_channel": local_config.get("home_channel"),
        "home_channel_kind": local_config.get("home_channel_kind"),
        "agent_number": identity.get("number"),
        "agent_name": identity.get("name"),
        "steps": [
            "Start the preferred command and keep it running for the receive window.",
            "From an allowed WhatsApp user, send a fresh voice note to the configured agent chat while the command is waiting.",
            "Use the fallback command only when Hermes is not consuming the bridge queue and it is acceptable to drain GET /messages.",
        ],
    }

    for item in components:
        if item["name"] not in receive_names:
            continue
        summary = item.get("summary") or {}
        checks = summary.get("checks") or {}
        inbound = checks.get("inbound_audio") or {}
        audio_events = inbound.get("audio_events") or []
        gate["component"] = item["name"]
        gate["drains_bridge_messages"] = bool(inbound.get("drains_bridge_messages"))
        if item.get("success") and audio_events:
            gate["status"] = "verified"
            gate["requires_operator"] = False
            gate["reason"] = "fresh WhatsApp inbound audio event observed"
            gate["audio_events"] = len(audio_events)
        elif item.get("success"):
            gate["status"] = "not_verified"
            gate["reason"] = "receive polling completed without fresh audio-event evidence"
            gate["audio_events"] = len(audio_events)
        else:
            gate["status"] = "failed"
            gate["reason"] = "fresh receive verifier failed"
            gate["failures"] = item.get("failures") or []
        break
    else:
        for item in components:
            if item["name"] != "whatsapp_inbound_cache_fresh_stt":
                continue
            summary = item.get("summary") or {}
            checks = summary.get("checks") or {}
            fresh_watch = checks.get("fresh_watch") or {}
            fresh_files = fresh_watch.get("fresh_files") or []
            gate["component"] = item["name"]
            gate["drains_bridge_messages"] = False
            if item.get("success") and fresh_files:
                gate["status"] = "verified"
                gate["requires_operator"] = False
                gate["reason"] = "fresh WhatsApp inbound audio cache artifact observed"
                gate["fresh_files"] = len(fresh_files)
            elif item.get("success"):
                gate["status"] = "not_verified"
                gate["reason"] = "audio cache watch completed without fresh audio evidence"
                gate["fresh_files"] = len(fresh_files)
            else:
                gate["status"] = "failed"
                gate["reason"] = "fresh audio cache receive verifier failed"
                gate["failures"] = item.get("failures") or []
            break
    return gate


def presence_sources(
    presence: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for key in keys:
        status = presence.get(key) if isinstance(presence, dict) else None
        if isinstance(status, dict):
            key_sources = status.get("sources") or []
            sources[key] = [str(source) for source in key_sources]
        else:
            sources[key] = []
    return sources


def source_key_inventory(bridge_checks: dict[str, Any]) -> dict[str, list[str]]:
    raw = bridge_checks.get("env_key_sources") if isinstance(bridge_checks, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {
        str(source): [str(key) for key in keys]
        for source, keys in raw.items()
        if isinstance(keys, list)
    }


def cloud_setup_handoff(
    *,
    external_meta_setup: dict[str, Any],
    bridge_checks: dict[str, Any],
    hermes_home: Path,
) -> dict[str, Any]:
    cloud = bridge_checks.get("whatsapp_cloud") or {}
    cloud_required = cloud.get("cloud_required") or {}
    missing = [str(key) for key in external_meta_setup.get("cloud_missing") or []]
    configured = bool(external_meta_setup.get("cloud_configured"))
    verify_command = [
        "scripts/verify_whatsapp_alpha_readiness.py",
        "--hermes-home",
        str(hermes_home),
        "--require-whatsapp-cloud",
    ]
    steps = [] if configured else [
        "Create or select the Meta app, WABA, and Cloud API phone number for Quill.",
        "Add the required WHATSAPP_CLOUD_* keys to the Hermes environment without committing secret values.",
        "Restart hermes-gateway.service so the gateway process sees the new Cloud credentials.",
        "Rerun the Cloud verification command and confirm whatsapp_cloud=configured.",
    ]
    return {
        "required_keys": list(CLOUD_REQUIRED_KEYS),
        "missing": missing,
        "configured": configured,
        "env_file": bridge_checks.get("env_file"),
        "credential_sources": presence_sources(cloud_required, CLOUD_REQUIRED_KEYS),
        "available_source_keys": source_key_inventory(bridge_checks),
        "verify_command": verify_command,
        "steps": steps,
    }


def calling_setup_handoff(
    *,
    external_meta_setup: dict[str, Any],
    bridge_checks: dict[str, Any],
    hermes_home: Path,
) -> dict[str, Any]:
    cloud = bridge_checks.get("whatsapp_cloud") or {}
    calling = cloud.get("calling") or {}
    cloud_required = cloud.get("cloud_required") or {}
    missing = [str(key) for key in external_meta_setup.get("calling_missing") or []]
    sidecar_missing = [
        key
        for key in CALLING_SIDECAR_REQUIRED_KEYS
        if key in missing
    ]
    ready = bool(external_meta_setup.get("calling_ready"))
    verify_command = [
        "scripts/verify_whatsapp_alpha_readiness.py",
        "--hermes-home",
        str(hermes_home),
        "--require-whatsapp-cloud",
        "--require-whatsapp-calling",
    ]
    complete_command = [
        "scripts/verify_whatsapp_alpha_readiness.py",
        "--hermes-home",
        str(hermes_home),
        "--profile",
        "attended-cache-receive",
        "--require-complete",
    ]
    steps = [] if ready else [
        "Complete WhatsApp Cloud setup first; Calling depends on the same Cloud phone credentials.",
        "Enable or approve WhatsApp Calling for the configured Cloud phone number in Meta.",
        "Ensure Hermes has the Calling sidecar URL and TTS stream command environment keys.",
        "Restart hermes-gateway.service and voice-webrtc-sidecar.service after environment or sidecar changes.",
        "Rerun the Calling verification command before using the complete alpha gate.",
    ]
    return {
        "required_keys": [*CLOUD_REQUIRED_KEYS, *CALLING_SIDECAR_REQUIRED_KEYS],
        "missing": missing,
        "cloud_missing": [
            key for key in missing if key in CLOUD_REQUIRED_KEYS
        ],
        "sidecar_missing": sidecar_missing,
        "calling_sidecar_configured": bool(
            external_meta_setup.get("calling_sidecar_configured")
        ),
        "calling_ready": ready,
        "env_file": bridge_checks.get("env_file"),
        "cloud_credential_sources": presence_sources(
            cloud_required,
            CLOUD_REQUIRED_KEYS,
        ),
        "sidecar_sources": presence_sources(
            calling,
            CALLING_SIDECAR_REQUIRED_KEYS,
        ),
        "available_source_keys": source_key_inventory(bridge_checks),
        "verify_command": verify_command,
        "complete_verification_command": complete_command,
        "steps": steps,
    }


def external_meta_gate(
    external_meta_setup: dict[str, Any],
    *,
    bridge_checks: dict[str, Any],
    hermes_home: Path,
) -> dict[str, Any]:
    cloud_missing = [str(key) for key in external_meta_setup.get("cloud_missing") or []]
    calling_missing = [
        str(key) for key in external_meta_setup.get("calling_missing") or []
    ]
    cloud_configured = bool(external_meta_setup.get("cloud_configured"))
    calling_ready = bool(external_meta_setup.get("calling_ready"))
    cloud_handoff = cloud_setup_handoff(
        external_meta_setup=external_meta_setup,
        bridge_checks=bridge_checks,
        hermes_home=hermes_home,
    )
    calling_handoff = calling_setup_handoff(
        external_meta_setup=external_meta_setup,
        bridge_checks=bridge_checks,
        hermes_home=hermes_home,
    )
    return {
        "whatsapp_cloud": {
            "status": "configured" if cloud_configured else "external_setup_required",
            "missing": cloud_missing,
            "setup_handoff": cloud_handoff,
        },
        "whatsapp_cloud_calling": {
            "status": "ready" if calling_ready else "external_setup_required",
            "missing": calling_missing,
            "setup_steps": (
                []
                if calling_ready
                else list(external_meta_setup.get("setup_steps") or META_SETUP_STEPS)
            ),
            "setup_handoff": calling_handoff,
        },
    }


def build_readiness_summary(
    *,
    local_checks_passed: bool,
    pending_gates: dict[str, Any],
    external_meta_setup: dict[str, Any],
) -> dict[str, Any]:
    attended = pending_gates.get("attended_fresh_receive") or {}
    attended_verified = attended.get("status") == "verified"
    external_meta_setup_required = not (
        external_meta_setup.get("cloud_configured")
        and external_meta_setup.get("calling_ready")
    )
    next_actions: list[dict[str, Any]] = []

    if not local_checks_passed:
        next_actions.append(
            {
                "id": "fix_local_runtime",
                "requires_operator": False,
                "description": "Fix failed local readiness components before retrying.",
            }
        )
    if not attended_verified:
        action: dict[str, Any] = {
            "id": "run_attended_fresh_receive",
            "requires_operator": True,
            "description": "Run the non-draining attended receive profile while someone sends a fresh WhatsApp voice note.",
        }
        command = attended.get("command")
        if command:
            action["command"] = command
        fallback = attended.get("fallback_draining_command")
        if fallback:
            action["fallback_draining_command"] = fallback
        next_actions.append(action)
    if external_meta_setup_required:
        next_actions.append(
            {
                "id": "configure_whatsapp_cloud_calling",
                "requires_operator": True,
                "description": "Complete external Meta WhatsApp Cloud and Calling setup, then rerun with Cloud/Calling requirements.",
                "missing": external_meta_setup.get("calling_missing") or [],
                "setup_steps": external_meta_setup.get("setup_steps") or [],
            }
        )

    complete = local_checks_passed and attended_verified and not external_meta_setup_required
    if complete:
        status = "complete"
    elif local_checks_passed:
        status = "local_ready_pending_gates"
    else:
        status = "local_checks_failed"

    return {
        "status": status,
        "complete": complete,
        "local_checks_passed": local_checks_passed,
        "attended_fresh_receive_verified": attended_verified,
        "external_meta_setup_required": external_meta_setup_required,
        "operator_action_required": any(
            bool(action.get("requires_operator")) for action in next_actions
        ),
        "next_actions": next_actions,
    }


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

    bridge_checks = bridge_runtime_checks(components)
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

    pending_gates = {
        "attended_fresh_receive": attended_receive_gate(
            components,
            hermes_home=args.hermes_home.expanduser().resolve(),
            audio_cache_dir=(
                args.whatsapp_audio_cache_dir.expanduser().resolve()
                if args.whatsapp_audio_cache_dir
                else args.hermes_home.expanduser().resolve() / "audio_cache"
            ),
        ),
        **external_meta_gate(
            external_meta_setup,
            bridge_checks=bridge_checks,
            hermes_home=args.hermes_home.expanduser().resolve(),
        ),
    }
    readiness_summary = build_readiness_summary(
        local_checks_passed=not required_failures,
        pending_gates=pending_gates,
        external_meta_setup=external_meta_setup,
    )
    completion_failures: list[dict[str, Any]] = []
    if args.require_complete and not readiness_summary["complete"]:
        success = False
        next_action_ids = [
            str(action.get("id"))
            for action in readiness_summary.get("next_actions") or []
            if action.get("id")
        ]
        failure_messages = [
            "WhatsApp alpha readiness is not complete: "
            + str(readiness_summary["status"])
        ]
        if next_action_ids:
            failure_messages.append(
                "next actions: " + ", ".join(next_action_ids)
            )
        completion_failures.append(
            {
                "name": "whatsapp_alpha_complete",
                "category": "readiness_summary",
                "failures": failure_messages,
            }
        )

    return {
        "success": success,
        "profile": args.profile,
        "components": components,
        "by_category": by_category,
        "external_meta_setup": external_meta_setup,
        "pending_gates": pending_gates,
        "readiness_summary": readiness_summary,
        "failures": [
            {
                "name": item["name"],
                "category": item["category"],
                "failures": item.get("failures") or [],
            }
            for item in required_failures
        ]
        + external_failures
        + completion_failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="unattended",
        help=(
            "named readiness profile: unattended dry run, cached receive STT, "
            "real send, non-draining attended cache receive, or attended "
            "bridge send/receive"
        ),
    )
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
    parser.add_argument(
        "--send-voice-note",
        action="store_true",
        help="post a real WhatsApp voice note through the local bridge",
    )
    parser.add_argument(
        "--voice-note-chat-id",
        default=None,
        help="override WHATSAPP_HOME_CHANNEL for the real voice-note send",
    )
    parser.add_argument("--wait-inbound-seconds", type=float, default=0.0)
    parser.add_argument("--require-inbound-audio", action="store_true")
    parser.add_argument(
        "--drain-bridge-messages",
        action="store_true",
        help=(
            "allow attended inbound receive polling via the bridge /messages "
            "endpoint; this consumes queued bridge messages"
        ),
    )
    parser.add_argument("--run-stt-smoke", action="store_true")
    parser.add_argument("--run-inbound-cache-smoke", action="store_true")
    parser.add_argument("--wait-audio-cache-seconds", type=float, default=0.0)
    parser.add_argument("--require-fresh-cache-audio", action="store_true")
    parser.add_argument("--require-whatsapp-cloud", action="store_true")
    parser.add_argument("--require-whatsapp-calling", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "fail unless local checks, attended fresh receive, and WhatsApp "
            "Cloud/Calling gates are all complete"
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def apply_profile(args: argparse.Namespace) -> None:
    if args.profile == "cached-receive":
        args.run_inbound_cache_smoke = True
    elif args.profile == "send":
        args.send_voice_note = True
    elif args.profile == "attended-cache-receive":
        args.send_voice_note = True
        args.run_inbound_cache_smoke = True
        if args.wait_audio_cache_seconds == 0:
            args.wait_audio_cache_seconds = 60.0
        args.require_fresh_cache_audio = True
    elif args.profile == "attended-send-receive":
        args.send_voice_note = True
        if args.wait_inbound_seconds == 0:
            args.wait_inbound_seconds = 60.0
        args.require_inbound_audio = True
        args.drain_bridge_messages = True


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    voice_note_options = [
        args.send_voice_note,
        args.voice_note_chat_id is not None,
        args.wait_inbound_seconds != 0,
        args.require_inbound_audio,
        args.drain_bridge_messages,
    ]
    if args.skip_voice_note_smoke and any(voice_note_options):
        parser.error(
            "voice-note send/receive flags cannot be used with --skip-voice-note-smoke"
        )
    if args.wait_inbound_seconds < 0:
        parser.error("--wait-inbound-seconds must be non-negative")
    if args.wait_audio_cache_seconds < 0:
        parser.error("--wait-audio-cache-seconds must be non-negative")
    if args.voice_note_chat_id and not args.send_voice_note:
        parser.error("--voice-note-chat-id requires --send-voice-note")
    if args.require_inbound_audio and args.wait_inbound_seconds <= 0:
        parser.error("--require-inbound-audio requires --wait-inbound-seconds")
    if args.require_fresh_cache_audio and args.wait_audio_cache_seconds <= 0:
        parser.error(
            "--require-fresh-cache-audio requires --wait-audio-cache-seconds"
        )
    if args.drain_bridge_messages and args.wait_inbound_seconds <= 0:
        parser.error("--drain-bridge-messages requires --wait-inbound-seconds")
    if args.wait_inbound_seconds > 0 and not args.drain_bridge_messages:
        parser.error(
            "--wait-inbound-seconds polls the bridge /messages queue; "
            "add --drain-bridge-messages only during an attended receive test"
        )


def human_summary(result: dict[str, Any]) -> None:
    if result["success"]:
        print("ok: WhatsApp alpha readiness passed")
    else:
        print("error: WhatsApp alpha readiness failed", file=sys.stderr)
    print(f"profile={result.get('profile')}")
    summary = result.get("readiness_summary") or {}
    if summary:
        print(
            "readiness="
            f"{summary.get('status')} complete={summary.get('complete')} "
            f"operator_action_required={summary.get('operator_action_required')} "
            f"external_meta_setup_required={summary.get('external_meta_setup_required')}"
        )

    for item in result["components"]:
        status = "ok" if item["success"] else "failed"
        print(f"{item['name']}={status} category={item['category']}")
        if not item["success"]:
            for failure in item.get("failures") or []:
                print(f"  - {failure}", file=sys.stderr)

    component_names = {item["name"] for item in result["components"]}
    for failure in result.get("failures") or []:
        if failure.get("name") in component_names:
            continue
        print(
            f"{failure.get('name')}=failed category={failure.get('category')}"
        )
        for message in failure.get("failures") or []:
            print(f"  - {message}", file=sys.stderr)

    external = result["external_meta_setup"]
    pending = result.get("pending_gates") or {}
    attended = pending.get("attended_fresh_receive") or {}
    if attended:
        print(
            "attended_fresh_receive="
            f"{attended.get('status')} cached_receive_verified="
            f"{attended.get('cached_receive_verified')}"
        )
        if attended.get("status") != "verified":
            command = attended.get("command") or []
            fallback = attended.get("fallback_draining_command") or []
            if command:
                print(
                    "attended_fresh_receive_command="
                    f"{shlex.join([str(part) for part in command])}"
                )
            if fallback:
                print(
                    "attended_fresh_receive_fallback_draining_command="
                    f"{shlex.join([str(part) for part in fallback])}"
                )
            handoff = attended.get("operator_handoff") or {}
            if handoff:
                print(
                    "attended_fresh_receive_operator="
                    f"agent={handoff.get('agent_name') or '<unknown>'} "
                    f"number={handoff.get('agent_number') or '<unknown>'} "
                    f"home_channel={handoff.get('home_channel') or '<unknown>'} "
                    f"audio_cache_dir={handoff.get('audio_cache_dir')}"
                )
                for index, step in enumerate(handoff.get("steps") or [], start=1):
                    print(f"attended_fresh_receive_step[{index}]={step}")
    print(
        "whatsapp_cloud="
        + ("configured" if external["cloud_configured"] else "not_configured")
        + " missing="
        + (",".join(external["cloud_missing"]) or "none")
    )
    cloud_gate = pending.get("whatsapp_cloud") or {}
    cloud_handoff = cloud_gate.get("setup_handoff") or {}
    if cloud_gate.get("status") != "configured" and cloud_handoff:
        print(
            "whatsapp_cloud_setup="
            f"env_file={cloud_handoff.get('env_file') or '<unknown>'} "
            "missing="
            + (",".join(cloud_handoff.get("missing") or []) or "none")
        )
        verify_command = cloud_handoff.get("verify_command") or []
        if verify_command:
            print(
                "whatsapp_cloud_verify_command="
                f"{shlex.join([str(part) for part in verify_command])}"
            )
        for index, step in enumerate(cloud_handoff.get("steps") or [], start=1):
            print(f"whatsapp_cloud_step[{index}]={step}")
    print(
        "whatsapp_calling="
        + ("ready" if external["calling_ready"] else "not_ready")
        + " missing="
        + (",".join(external["calling_missing"]) or "none")
    )
    calling_gate = pending.get("whatsapp_cloud_calling") or {}
    calling_handoff = calling_gate.get("setup_handoff") or {}
    if calling_gate.get("status") != "ready" and calling_handoff:
        print(
            "whatsapp_calling_setup="
            f"sidecar_configured={calling_handoff.get('calling_sidecar_configured')} "
            "missing="
            + (",".join(calling_handoff.get("missing") or []) or "none")
        )
        verify_command = calling_handoff.get("verify_command") or []
        complete_command = calling_handoff.get("complete_verification_command") or []
        if verify_command:
            print(
                "whatsapp_calling_verify_command="
                f"{shlex.join([str(part) for part in verify_command])}"
            )
        if complete_command:
            print(
                "whatsapp_calling_complete_command="
                f"{shlex.join([str(part) for part in complete_command])}"
            )
        for index, step in enumerate(calling_handoff.get("steps") or [], start=1):
            print(f"whatsapp_calling_step[{index}]={step}")
    if external["setup_steps"]:
        print("external_meta_setup=required")
        for step in external["setup_steps"]:
            print(f"- {step}")
    else:
        print("external_meta_setup=complete")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_profile(args)
    validate_args(parser, args)
    result = build_readiness(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        human_summary(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
