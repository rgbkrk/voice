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
DEFAULT_ATTENDED_PROMPT_TEXT = (
    "Please reply with a fresh WhatsApp voice note so I can verify the voice runtime."
)
ATTENDED_PROFILE = "attended-cache-receive"


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


def utc_iso_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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
        ATTENDED_PROFILE,
        "--attended-prompt-text",
        args.attended_prompt_text,
        "--wait-audio-cache-seconds",
        str(float(args.wait_seconds)),
    ]
    if args.expected_agent_number:
        command.extend(["--expected-agent-number", args.expected_agent_number])
    if args.expected_agent_name:
        command.extend(["--expected-agent-name", args.expected_agent_name])
    command.append("--json")
    return command


def attended_prompt_manifest(args: argparse.Namespace) -> dict[str, Any]:
    audio_cache_dir = args.hermes_home / "audio_cache"
    return {
        "sends_prompt_voice_note": True,
        "prompt_text": args.attended_prompt_text,
        "send_profile": ATTENDED_PROFILE,
        "send_format": "audio/ogg; codecs=opus",
        "send_transport": "local_whatsapp_bridge_ptt",
        "receive_watch": "non_draining_audio_cache",
        "audio_cache_dir": str(audio_cache_dir),
        "operator_action": (
            "Send a fresh WhatsApp voice note to the configured agent chat "
            "while this watch is running."
        ),
    }


def build_watch(args: argparse.Namespace) -> dict[str, Any]:
    unit = watch_unit_name(args)
    service = service_name(unit)
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / f"{unit}.json"
    log_path = output_dir / f"{unit}.log"
    manifest_path = output_dir / f"{unit}.manifest.json"
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
        "manifest_path": str(manifest_path),
        "alpha_command": alpha_command,
        "systemd_command": systemd_command,
        "status_command": ["systemctl", "--user", "status", service],
        "journal_command": ["journalctl", "--user", "-u", service, "-f"],
    }


def build_manifest(args: argparse.Namespace, watch: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "voice.whatsapp_attended_cache_watch_manifest",
        "version": 1,
        "profile": ATTENDED_PROFILE,
        "drains_bridge_messages": False,
        "created_at_utc": utc_iso_timestamp(),
        "unit": watch["unit"],
        "service": watch["service"],
        "wait_seconds": float(args.wait_seconds),
        "attended_prompt": attended_prompt_manifest(args),
        "voice_bin": str(args.voice_bin),
        "hermes_home": str(args.hermes_home),
        "hermes_config": str(args.hermes_config),
        "expected_agent_number": args.expected_agent_number or None,
        "expected_agent_name": args.expected_agent_name or None,
        "artifacts": {
            "json": watch["json_path"],
            "log": watch["log_path"],
            "manifest": watch["manifest_path"],
        },
        "commands": {
            "alpha": watch["alpha_command"],
            "systemd": watch["systemd_command"],
            "status": watch["status_command"],
            "journal": watch["journal_command"],
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def service_name(unit: str) -> str:
    return unit if unit.endswith(".service") else f"{unit}.service"


def watch_unit_name(args: argparse.Namespace) -> str:
    timestamp = args.timestamp or utc_timestamp()
    return f"{args.unit_prefix}-{timestamp}"


def normalize_status_unit(value: str) -> tuple[str, str]:
    service = service_name(value)
    unit = service.removesuffix(".service")
    return unit, service


def parse_systemctl_show(output: str) -> dict[str, str]:
    state: dict[str, str] = {}
    for line in output.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key] = value
    return state


def artifact_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": 0,
        "parsed": False,
        "parse_error": None,
        "payload": None,
    }
    if not path.exists():
        return status
    status["size_bytes"] = path.stat().st_size
    if status["size_bytes"] <= 0:
        return status
    try:
        status["payload"] = json.loads(path.read_text(encoding="utf-8"))
        status["parsed"] = True
    except (OSError, json.JSONDecodeError) as exc:
        status["parse_error"] = str(exc)
    return status


def summarize_alpha_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("readiness_summary")
    summary = summary if isinstance(summary, dict) else {}
    pending = payload.get("pending_gates")
    pending = pending if isinstance(pending, dict) else {}
    attended = pending.get("attended_fresh_receive")
    attended = attended if isinstance(attended, dict) else {}
    evidence = attended.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "profile": payload.get("profile"),
        "success": bool(payload.get("success")),
        "readiness_status": summary.get("status"),
        "complete": summary.get("complete"),
        "attended_fresh_receive_verified": summary.get(
            "attended_fresh_receive_verified"
        ),
        "external_meta_setup_required": summary.get("external_meta_setup_required"),
        "attended_status": attended.get("status"),
        "cached_receive_verified": attended.get("cached_receive_verified"),
        "evidence_kind": evidence.get("kind"),
        "fresh_count": evidence.get("fresh_count"),
    }


def summarize_manifest_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    artifacts = payload.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    return {
        "schema": payload.get("schema"),
        "version": payload.get("version"),
        "profile": payload.get("profile"),
        "created_at_utc": payload.get("created_at_utc"),
        "wait_seconds": payload.get("wait_seconds"),
        "drains_bridge_messages": payload.get("drains_bridge_messages"),
        "expected_agent_number": payload.get("expected_agent_number"),
        "expected_agent_name": payload.get("expected_agent_name"),
        "voice_bin": payload.get("voice_bin"),
        "hermes_home": payload.get("hermes_home"),
        "hermes_config": payload.get("hermes_config"),
        "json_path": artifacts.get("json"),
        "log_path": artifacts.get("log"),
        "manifest_path": artifacts.get("manifest"),
        "attended_prompt": payload.get("attended_prompt"),
    }


def classify_watch_status(
    *,
    systemd_state: dict[str, str],
    json_artifact: dict[str, Any],
    alpha_summary: dict[str, Any],
) -> str:
    active_state = systemd_state.get("ActiveState")
    if active_state == "active":
        if json_artifact.get("parsed"):
            return "completed_running"
        return "waiting_for_fresh_audio"
    if json_artifact.get("parsed"):
        if alpha_summary.get("attended_status") == "verified":
            return "verified"
        if alpha_summary.get("success"):
            return "completed"
        return "failed"
    if json_artifact.get("exists"):
        return "empty_artifact"
    return "no_artifact"


def inspect_unit(unit_or_service: str, args: argparse.Namespace) -> dict[str, Any]:
    unit, service = normalize_status_unit(unit_or_service)
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / f"{unit}.json"
    log_path = output_dir / f"{unit}.log"
    manifest_path = output_dir / f"{unit}.manifest.json"
    systemctl_command = [
        str(args.systemctl_bin),
        "--user",
        "show",
        service,
        "-p",
        "ActiveState",
        "-p",
        "SubState",
        "-p",
        "MainPID",
    ]
    completed = subprocess.run(
        systemctl_command,
        check=False,
        text=True,
        capture_output=True,
    )
    systemd_state = parse_systemctl_show(completed.stdout)
    json_artifact = artifact_status(json_path)
    log_artifact = artifact_status(log_path)
    log_artifact.pop("payload", None)
    manifest_artifact = artifact_status(manifest_path)
    manifest_summary = summarize_manifest_payload(manifest_artifact.get("payload"))
    manifest_artifact.pop("payload", None)
    alpha_summary = summarize_alpha_payload(json_artifact.get("payload"))
    watch_status = classify_watch_status(
        systemd_state=systemd_state,
        json_artifact=json_artifact,
        alpha_summary=alpha_summary,
    )
    json_artifact.pop("payload", None)
    return {
        "unit": unit,
        "service": service,
        "watch_status": watch_status,
        "systemctl_command": systemctl_command,
        "systemctl_returncode": completed.returncode,
        "systemctl_stderr": completed.stderr.strip(),
        "systemd": systemd_state,
        "json": json_artifact,
        "log": log_artifact,
        "manifest": manifest_artifact,
        "manifest_summary": manifest_summary,
        "alpha": alpha_summary,
        "status_command": ["systemctl", "--user", "status", service],
        "journal_command": ["journalctl", "--user", "-u", service, "-f"],
    }


def inspect_status(args: argparse.Namespace) -> dict[str, Any]:
    return inspect_unit(args.status, args)


def list_systemd_watch_units(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.systemctl_bin),
        "--user",
        "list-units",
        "--type=service",
        "--all",
        "--no-legend",
        "--no-pager",
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    units: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        service = parts[0]
        unit = service.removesuffix(".service")
        if unit.startswith(f"{args.unit_prefix}-"):
            units.append(unit)
    return units


def list_artifact_watch_units(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir.expanduser().resolve()
    units: set[str] = set()
    for pattern in (f"{args.unit_prefix}-*.json", f"{args.unit_prefix}-*.log"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                unit = unit_from_artifact_path(path)
                if unit:
                    units.add(unit)
    return sorted(units)


def unit_from_artifact_path(path: Path) -> str | None:
    name = path.name
    if name.endswith(".manifest.json"):
        return name[: -len(".manifest.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    if name.endswith(".log"):
        return name[: -len(".log")]
    return None


def list_watches(args: argparse.Namespace) -> dict[str, Any]:
    units = sorted(
        set(list_systemd_watch_units(args)) | set(list_artifact_watch_units(args)),
        reverse=True,
    )
    watches = [inspect_unit(unit, args) for unit in units]
    return {
        "unit_prefix": args.unit_prefix,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "count": len(watches),
        "watches": watches,
    }


def stop_watch(args: argparse.Namespace) -> dict[str, Any]:
    unit, service = normalize_status_unit(args.stop)
    command = [str(args.systemctl_bin), "--user", "stop", service]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    status = inspect_unit(unit, args)
    return {
        **status,
        "stop_command": command,
        "stop_returncode": completed.returncode,
        "stop_stdout": completed.stdout.strip(),
        "stop_stderr": completed.stderr.strip(),
    }


def validate_args(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    modes = [bool(args.status), bool(args.list), bool(args.stop)]
    if sum(modes) > 1:
        failures.append("--status, --list, and --stop are mutually exclusive")
    if not any(modes) and args.wait_seconds <= 0:
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
    print(f"manifest={result['manifest_path']}")
    print(f"wait_seconds={result['wait_seconds']}")
    attended_prompt = (result.get("manifest") or {}).get("attended_prompt") or {}
    if attended_prompt:
        print(f"sends_prompt_voice_note={attended_prompt.get('sends_prompt_voice_note')}")
        print(f"prompt_text={attended_prompt.get('prompt_text')}")
        print(f"audio_cache_dir={attended_prompt.get('audio_cache_dir')}")
        print(f"operator_action={attended_prompt.get('operator_action')}")
    print(f"alpha_command={shlex.join(result['alpha_command'])}")
    print(f"systemd_command={shlex.join(result['systemd_command'])}")
    print(f"status_command={shlex.join(result['status_command'])}")
    print(f"journal_command={shlex.join(result['journal_command'])}")


def print_status_human(result: dict[str, Any]) -> None:
    print("ok: WhatsApp attended cache watch status")
    print(f"unit={result['unit']}")
    print(f"service={result['service']}")
    print(f"watch_status={result['watch_status']}")
    print(f"active_state={result.get('systemd', {}).get('ActiveState')}")
    print(f"sub_state={result.get('systemd', {}).get('SubState')}")
    print(f"main_pid={result.get('systemd', {}).get('MainPID')}")
    json_artifact = result.get("json") or {}
    log_artifact = result.get("log") or {}
    manifest_artifact = result.get("manifest") or {}
    print(f"json={json_artifact.get('path')} size={json_artifact.get('size_bytes')}")
    print(f"log={log_artifact.get('path')} size={log_artifact.get('size_bytes')}")
    print(
        f"manifest={manifest_artifact.get('path')} "
        f"size={manifest_artifact.get('size_bytes')}"
    )
    manifest_summary = result.get("manifest_summary") or {}
    if manifest_summary:
        print(
            "manifest_summary="
            f"profile={manifest_summary.get('profile')} "
            f"created_at_utc={manifest_summary.get('created_at_utc')} "
            f"wait_seconds={manifest_summary.get('wait_seconds')} "
            "drains_bridge_messages="
            f"{manifest_summary.get('drains_bridge_messages')} "
            f"expected_agent_number={manifest_summary.get('expected_agent_number')} "
            f"expected_agent_name={manifest_summary.get('expected_agent_name')}"
        )
        attended_prompt = manifest_summary.get("attended_prompt")
        attended_prompt = attended_prompt if isinstance(attended_prompt, dict) else {}
        if attended_prompt:
            print(
                "attended_prompt="
                f"sends_prompt_voice_note="
                f"{attended_prompt.get('sends_prompt_voice_note')} "
                f"prompt_text={attended_prompt.get('prompt_text')} "
                f"audio_cache_dir={attended_prompt.get('audio_cache_dir')}"
            )
    alpha = result.get("alpha") or {}
    if alpha:
        print(
            "alpha="
            f"profile={alpha.get('profile')} "
            f"success={alpha.get('success')} "
            f"readiness={alpha.get('readiness_status')} "
            f"attended_status={alpha.get('attended_status')} "
            "attended_fresh_receive_verified="
            f"{alpha.get('attended_fresh_receive_verified')} "
            "external_meta_setup_required="
            f"{alpha.get('external_meta_setup_required')}"
        )
    print(f"status_command={shlex.join(result['status_command'])}")
    print(f"journal_command={shlex.join(result['journal_command'])}")


def print_stop_human(result: dict[str, Any]) -> None:
    if result.get("stop_returncode") == 0:
        print("ok: WhatsApp attended cache watch stop requested")
    else:
        print("error: WhatsApp attended cache watch stop failed", file=sys.stderr)
    print(f"unit={result['unit']}")
    print(f"service={result['service']}")
    print(f"stop_returncode={result['stop_returncode']}")
    if result.get("stop_stderr"):
        print(f"stop_stderr={result['stop_stderr']}")
    print(f"watch_status={result['watch_status']}")
    print(f"active_state={result.get('systemd', {}).get('ActiveState')}")
    print(f"sub_state={result.get('systemd', {}).get('SubState')}")
    json_artifact = result.get("json") or {}
    log_artifact = result.get("log") or {}
    manifest_artifact = result.get("manifest") or {}
    print(f"json={json_artifact.get('path')} size={json_artifact.get('size_bytes')}")
    print(f"log={log_artifact.get('path')} size={log_artifact.get('size_bytes')}")
    print(
        f"manifest={manifest_artifact.get('path')} "
        f"size={manifest_artifact.get('size_bytes')}"
    )


def print_list_human(result: dict[str, Any]) -> None:
    print("ok: WhatsApp attended cache watch list")
    print(f"unit_prefix={result['unit_prefix']}")
    print(f"output_dir={result['output_dir']}")
    print(f"count={result['count']}")
    for index, watch in enumerate(result.get("watches") or [], start=1):
        json_artifact = watch.get("json") or {}
        manifest_summary = watch.get("manifest_summary") or {}
        print(
            f"watch[{index}]={watch.get('unit')} "
            f"status={watch.get('watch_status')} "
            f"active_state={(watch.get('systemd') or {}).get('ActiveState')} "
            f"json_size={json_artifact.get('size_bytes')} "
            f"manifest_wait_seconds={manifest_summary.get('wait_seconds')} "
            "sends_prompt_voice_note="
            f"{(manifest_summary.get('attended_prompt') or {}).get('sends_prompt_voice_note')}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        metavar="UNIT",
        default=None,
        help="inspect a started attended watch unit instead of starting a new one",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list attended watch units and artifacts matching --unit-prefix",
    )
    parser.add_argument(
        "--stop",
        metavar="UNIT",
        default=None,
        help="stop a started attended watch unit without deleting artifacts",
    )
    parser.add_argument("--voice-bin", default=default_voice_bin())
    parser.add_argument("--hermes-home", type=Path, default=default_hermes_home())
    parser.add_argument("--hermes-config", type=Path, default=default_hermes_config())
    parser.add_argument("--wait-seconds", type=float, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--attended-prompt-text", default=DEFAULT_ATTENDED_PROMPT_TEXT)
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
    parser.add_argument(
        "--systemd-run-bin",
        default=os.environ.get("SYSTEMD_RUN_BIN", "systemd-run"),
    )
    parser.add_argument("--systemctl-bin", default=os.environ.get("SYSTEMCTL_BIN", "systemctl"))
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

    if args.status:
        result = inspect_status(args)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_status_human(result)
        return 0
    if args.list:
        result = list_watches(args)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_list_human(result)
        return 0
    if args.stop:
        result = stop_watch(args)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_stop_human(result)
        return int(result.get("stop_returncode") or 0)

    watch = build_watch(args)
    manifest = build_manifest(args, watch)
    result = {
        **watch,
        "dry_run": args.dry_run,
        "wait_seconds": args.wait_seconds,
        "manifest": manifest,
        "returncode": 0,
    }
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_manifest(Path(watch["manifest_path"]), manifest)
        except OSError as exc:
            result["returncode"] = 1
            result["stderr"] = f"failed to write manifest: {exc}"
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print_human(result)
                print(result["stderr"], file=sys.stderr)
            return 1
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
