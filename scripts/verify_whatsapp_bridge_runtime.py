#!/usr/bin/env python3
"""Verify the local WhatsApp bridge identity without exposing credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BRIDGE_URL = "http://127.0.0.1:3000"
DEFAULT_SERVICE_NAME = "hermes-gateway.service"

WHATSAPP_CLOUD_REQUIRED_ENV = (
    "WHATSAPP_CLOUD_PHONE_NUMBER_ID",
    "WHATSAPP_CLOUD_ACCESS_TOKEN",
    "WHATSAPP_CLOUD_APP_SECRET",
    "WHATSAPP_CLOUD_VERIFY_TOKEN",
)
WHATSAPP_CLOUD_OPTIONAL_ENV = (
    "WHATSAPP_CLOUD_APP_ID",
    "WHATSAPP_CLOUD_WABA_ID",
    "WHATSAPP_CLOUD_ALLOWED_USERS",
    "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
    "WHATSAPP_CLOUD_HOME_CHANNEL",
    "WHATSAPP_CLOUD_WEBHOOK_HOST",
    "WHATSAPP_CLOUD_WEBHOOK_PORT",
    "WHATSAPP_CLOUD_WEBHOOK_PATH",
    "WHATSAPP_CLOUD_API_VERSION",
)
WHATSAPP_CALLING_ENV = (
    "WHATSAPP_CLOUD_CALLING_SIDECAR_URL",
    "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND",
    "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_TIMEOUT",
)


def default_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def normalize_whatsapp_identifier(value: Any) -> str | None:
    if value is None:
        return None
    local = str(value).strip()
    if not local:
        return None
    local = local.split("@", 1)[0]
    local = local.split(":", 1)[0]
    digits = re.sub(r"\D", "", local)
    return digits or None


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        env[key] = value
    return env


def parse_systemd_environment(value: str) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        parts = shlex.split(value)
    except ValueError:
        return env
    for part in parts:
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        env[key] = raw
    return env


def get_systemd_service_env(
    service_name: str,
    *,
    timeout: float,
) -> tuple[dict[str, str], str | None]:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service_name,
            "-p",
            "Environment",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {}, detail or f"systemctl show failed for {service_name}"

    for line in completed.stdout.splitlines():
        if line.startswith("Environment="):
            return parse_systemd_environment(line.split("=", 1)[1]), None
    return {}, None


def get_systemd_main_pid(
    service_name: str,
    *,
    timeout: float,
) -> tuple[int | None, str | None]:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service_name,
            "-p",
            "MainPID",
            "--no-pager",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return None, detail or f"systemctl show MainPID failed for {service_name}"

    for line in completed.stdout.splitlines():
        if not line.startswith("MainPID="):
            continue
        raw_pid = line.split("=", 1)[1].strip()
        try:
            pid = int(raw_pid)
        except ValueError:
            return None, f"{service_name} MainPID is not an integer: {raw_pid!r}"
        return (pid if pid > 0 else None), None
    return None, None


def read_process_environment(pid: int) -> tuple[dict[str, str], str | None]:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError as exc:
        return {}, f"could not read /proc/{pid}/environ: {exc}"

    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        decoded_key = key.decode("utf-8", errors="replace")
        if not decoded_key:
            continue
        env[decoded_key] = value.decode("utf-8", errors="replace")
    return env, None


def read_json_file(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"file not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path}: {exc}"
    except OSError as exc:
        return None, f"could not read {path}: {exc}"


def session_artifact_counts(session_dir: Path) -> dict[str, int]:
    counts = {
        "pre_key": 0,
        "session": 0,
        "app_state_sync_key": 0,
        "lid_mapping": 0,
        "tctoken": 0,
        "sender_key": 0,
        "other_json": 0,
    }
    if not session_dir.is_dir():
        return counts

    for path in session_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("pre-key-") and name.endswith(".json"):
            counts["pre_key"] += 1
        elif name.startswith("session-") and name.endswith(".json"):
            counts["session"] += 1
        elif name.startswith("app-state-sync-key-") and name.endswith(".json"):
            counts["app_state_sync_key"] += 1
        elif name.startswith("lid-mapping-") and name.endswith(".json"):
            counts["lid_mapping"] += 1
        elif name.startswith("tctoken-") and name.endswith(".json"):
            counts["tctoken"] += 1
        elif name.startswith("sender-key-") and name.endswith(".json"):
            counts["sender_key"] += 1
        elif name.endswith(".json") and name != "creds.json":
            counts["other_json"] += 1
    return counts


def extract_identity(creds: dict[str, Any]) -> dict[str, Any]:
    me = creds.get("me") if isinstance(creds.get("me"), dict) else {}
    raw_id = me.get("id")
    raw_lid = me.get("lid")
    return {
        "name": me.get("name"),
        "id": raw_id,
        "number": normalize_whatsapp_identifier(raw_id),
        "lid": raw_lid,
        "lid_number": normalize_whatsapp_identifier(raw_lid),
        "platform": creds.get("platform"),
        "account_sync_counter": creds.get("accountSyncCounter"),
    }


def read_lid_json(path: Path) -> str | None:
    payload, _ = read_json_file(path)
    if payload is None:
        return None
    return normalize_whatsapp_identifier(payload)


def verify_lid_mapping(
    session_dir: Path,
    *,
    phone_number: str | None,
    lid_number: str | None,
) -> dict[str, Any]:
    if not phone_number or not lid_number:
        return {
            "checked": False,
            "ok": False,
            "reason": "missing phone number or LID in Baileys credentials",
        }

    forward_path = session_dir / f"lid-mapping-{phone_number}.json"
    reverse_path = session_dir / f"lid-mapping-{lid_number}_reverse.json"
    forward = read_lid_json(forward_path)
    reverse = read_lid_json(reverse_path)
    return {
        "checked": True,
        "ok": forward == lid_number and reverse == phone_number,
        "forward_path": str(forward_path),
        "reverse_path": str(reverse_path),
        "forward_present": forward is not None,
        "reverse_present": reverse is not None,
    }


def option_value(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name and index + 1 < len(args):
            return args[index + 1]
    return None


def find_bridge_processes(*, port: int | None = None) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return processes

    for child in proc.iterdir():
        if not child.name.isdigit():
            continue
        try:
            raw = (child / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if not any(arg.endswith("bridge.js") and "whatsapp-bridge" in arg for arg in argv):
            continue
        process_port = option_value(argv, "--port")
        if port is not None and process_port is not None:
            try:
                if int(process_port) != port:
                    continue
            except ValueError:
                continue
        script_path = next(
            (
                arg
                for arg in argv
                if arg.endswith("bridge.js") and "whatsapp-bridge" in arg
            ),
            None,
        )
        processes.append(
            {
                "pid": int(child.name),
                "script": script_path,
                "port": process_port,
                "session": option_value(argv, "--session"),
                "mode": option_value(argv, "--mode"),
            }
        )
    return processes


def fetch_bridge_health(bridge_url: str, *, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    url = bridge_url.rstrip("/") + "/health"
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        return None, f"bridge health returned HTTP {exc.code}"
    except URLError as exc:
        return None, f"bridge health request failed: {exc.reason}"
    except OSError as exc:
        return None, f"bridge health request failed: {exc}"

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"bridge health did not return JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "bridge health JSON must be an object"
    return payload, None


def env_presence(env_sources: dict[str, dict[str, str]], keys: tuple[str, ...]) -> dict[str, Any]:
    key_status: dict[str, Any] = {}
    for key in keys:
        sources = [
            source
            for source, values in env_sources.items()
            if values.get(key) not in (None, "")
        ]
        key_status[key] = {
            "present": bool(sources),
            "sources": sources,
        }
    return key_status


def missing_env_keys(env_sources: dict[str, dict[str, str]], keys: tuple[str, ...]) -> list[str]:
    return [
        key
        for key, status in env_presence(env_sources, keys).items()
        if not status["present"]
    ]


def build_cloud_summary(env_sources: dict[str, dict[str, str]]) -> dict[str, Any]:
    required_presence = env_presence(env_sources, WHATSAPP_CLOUD_REQUIRED_ENV)
    optional_presence = env_presence(env_sources, WHATSAPP_CLOUD_OPTIONAL_ENV)
    calling_presence = env_presence(env_sources, WHATSAPP_CALLING_ENV)
    cloud_missing = [
        key for key, status in required_presence.items() if not status["present"]
    ]
    sidecar_missing = [
        key
        for key in (
            "WHATSAPP_CLOUD_CALLING_SIDECAR_URL",
            "WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND",
        )
        if not calling_presence[key]["present"]
    ]
    calling_missing = cloud_missing + sidecar_missing
    return {
        "cloud_configured": not cloud_missing,
        "cloud_missing": cloud_missing,
        "cloud_required": required_presence,
        "cloud_optional": optional_presence,
        "calling_sidecar_configured": not sidecar_missing,
        "calling_ready": not calling_missing,
        "calling_missing": calling_missing,
        "calling": calling_presence,
    }


def port_from_url(url: str) -> int | None:
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/:]+:(\d+)(?:/|$)", url)
    if not match:
        return None
    return int(match.group(1))


def verify(args: argparse.Namespace) -> dict[str, Any]:
    hermes_home = args.hermes_home.expanduser().resolve()
    session_dir = (
        args.session_dir.expanduser().resolve()
        if args.session_dir
        else hermes_home / "whatsapp" / "session"
    )
    env_file = args.env_file.expanduser().resolve() if args.env_file else hermes_home / ".env"
    bridge_url = args.bridge_url.rstrip("/")

    failures: list[str] = []
    warnings: list[str] = []

    env_sources = {"env_file": parse_env_file(env_file)}
    service_env_error = None
    if not args.skip_systemd:
        service_env, service_env_error = get_systemd_service_env(
            args.service_name,
            timeout=args.timeout,
        )
        env_sources["systemd_service"] = service_env
        if service_env_error:
            warnings.append(
                f"could not inspect {args.service_name} environment: {service_env_error}"
            )
        main_pid, main_pid_error = get_systemd_main_pid(
            args.service_name,
            timeout=args.timeout,
        )
        if main_pid_error:
            warnings.append(
                f"could not inspect {args.service_name} MainPID: {main_pid_error}"
            )
        elif main_pid:
            process_env, process_env_error = read_process_environment(main_pid)
            env_sources["systemd_process"] = process_env
            if process_env_error:
                warnings.append(
                    f"could not inspect {args.service_name} process environment: "
                    f"{process_env_error}"
                )

    expected_agent_number = normalize_whatsapp_identifier(args.expected_agent_number)
    expected_agent_name = args.expected_agent_name
    expected_mode = args.expected_mode or env_sources["env_file"].get("WHATSAPP_MODE")

    if not session_dir.is_dir():
        failures.append(f"WhatsApp session directory not found: {session_dir}")

    creds_path = session_dir / "creds.json"
    creds_payload, creds_error = read_json_file(creds_path)
    if creds_error:
        failures.append(f"Baileys credentials are not readable: {creds_error}")
        identity: dict[str, Any] = {}
    elif not isinstance(creds_payload, dict):
        failures.append(f"Baileys credentials must be a JSON object: {creds_path}")
        identity = {}
    else:
        identity = extract_identity(creds_payload)
        if not identity.get("number"):
            failures.append("Baileys credentials do not contain me.id")
        if expected_agent_number and identity.get("number") != expected_agent_number:
            failures.append(
                "Baileys paired number "
                f"{identity.get('number')!r} does not match expected "
                f"{expected_agent_number!r}"
            )
        if expected_agent_name and identity.get("name") != expected_agent_name:
            failures.append(
                "Baileys paired name "
                f"{identity.get('name')!r} does not match expected "
                f"{expected_agent_name!r}"
            )

    counts = session_artifact_counts(session_dir)
    if counts["session"] <= 0:
        failures.append("Baileys session has no peer session files")
    if counts["pre_key"] <= 0:
        failures.append("Baileys session has no pre-key files")

    lid_mapping = verify_lid_mapping(
        session_dir,
        phone_number=identity.get("number"),
        lid_number=identity.get("lid_number"),
    )
    if lid_mapping["checked"] and not lid_mapping["ok"]:
        failures.append("Baileys LID mapping does not match the paired identity")
    elif not lid_mapping["checked"]:
        warnings.append(str(lid_mapping["reason"]))

    if args.skip_bridge_health:
        bridge_health = {"skipped": True}
    else:
        bridge_health_payload, bridge_health_error = fetch_bridge_health(
            bridge_url,
            timeout=args.timeout,
        )
        if bridge_health_error:
            failures.append(bridge_health_error)
            bridge_health = {}
        else:
            bridge_health = bridge_health_payload or {}
            if bridge_health.get("status") != "connected":
                failures.append(
                    f"WhatsApp bridge status={bridge_health.get('status')!r}, "
                    "expected 'connected'"
                )

    bridge_port = port_from_url(bridge_url)
    if args.skip_process_check:
        bridge_process = {"skipped": True}
    else:
        candidates = find_bridge_processes(port=bridge_port)
        bridge_process = {"candidates": candidates}
        if not candidates:
            failures.append("no running whatsapp-bridge/bridge.js process found")
        else:
            matching_session = [
                process
                for process in candidates
                if process.get("session")
                and Path(str(process["session"])).expanduser().resolve() == session_dir
            ]
            selected = matching_session[0] if matching_session else candidates[0]
            bridge_process["selected"] = selected
            process_session = selected.get("session")
            if process_session:
                resolved = Path(str(process_session)).expanduser().resolve()
                if resolved != session_dir:
                    failures.append(
                        f"bridge process session={resolved}, expected {session_dir}"
                    )
            else:
                warnings.append("bridge process does not expose --session in argv")
            process_mode = selected.get("mode")
            if expected_mode and process_mode and process_mode != expected_mode:
                failures.append(
                    f"bridge process mode={process_mode!r}, expected {expected_mode!r}"
                )

    cloud = build_cloud_summary(env_sources)
    if args.require_whatsapp_cloud and not cloud["cloud_configured"]:
        failures.append(
            "WhatsApp Cloud credentials missing: "
            + ", ".join(cloud["cloud_missing"])
        )
    if args.require_whatsapp_calling and not cloud["calling_ready"]:
        failures.append(
            "WhatsApp Cloud Calling not ready; missing: "
            + ", ".join(cloud["calling_missing"])
        )

    checks: dict[str, Any] = {
        "hermes_home": str(hermes_home),
        "env_file": str(env_file),
        "bridge_url": bridge_url,
        "session_dir": str(session_dir),
        "expected_agent_number": expected_agent_number,
        "expected_agent_name": expected_agent_name,
        "expected_mode": expected_mode,
        "baileys_identity": identity,
        "session_artifacts": counts,
        "lid_mapping": lid_mapping,
        "bridge_health": bridge_health,
        "bridge_process": bridge_process,
        "env_key_sources": {
            source: sorted(key for key in values if key.startswith("WHATSAPP"))
            for source, values in env_sources.items()
        },
        "whatsapp_cloud": cloud,
    }
    return {
        "success": not failures,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", type=Path, default=default_hermes_home())
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--bridge-url", default=os.environ.get("WHATSAPP_BRIDGE_URL", DEFAULT_BRIDGE_URL))
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--expected-agent-number", default=os.environ.get("WHATSAPP_AGENT_NUMBER"))
    parser.add_argument("--expected-agent-name", default=os.environ.get("WHATSAPP_AGENT_NAME"))
    parser.add_argument("--expected-mode", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--skip-bridge-health", action="store_true")
    parser.add_argument("--skip-process-check", action="store_true")
    parser.add_argument("--skip-systemd", action="store_true")
    parser.add_argument("--require-whatsapp-cloud", action="store_true")
    parser.add_argument("--require-whatsapp-calling", action="store_true")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def human_summary(result: dict[str, Any]) -> None:
    checks = result["checks"]
    identity = checks.get("baileys_identity") or {}
    health = checks.get("bridge_health") or {}
    process = (checks.get("bridge_process") or {}).get("selected") or {}
    cloud = checks.get("whatsapp_cloud") or {}

    if result["success"]:
        print("ok: WhatsApp bridge runtime verifier passed")
    else:
        print("error: WhatsApp bridge runtime verifier failed", file=sys.stderr)
        for failure in result["failures"]:
            print(f"- {failure}", file=sys.stderr)

    print(
        "bridge="
        f"{checks.get('bridge_url')} status={health.get('status', 'unknown')} "
        f"queue={health.get('queueLength', 'unknown')}"
    )
    print(
        "baileys_session=paired "
        f"name={identity.get('name') or '<unknown>'} "
        f"number={identity.get('number') or '<unknown>'} "
        f"lid={identity.get('lid_number') or '<unknown>'} "
        f"platform={identity.get('platform') or '<unknown>'}"
    )
    if process:
        print(
            "bridge_process="
            f"pid={process.get('pid')} mode={process.get('mode') or '<unknown>'} "
            f"session={process.get('session') or '<unknown>'}"
        )
    print(
        "session_artifacts="
        + ",".join(
            f"{key}={value}"
            for key, value in sorted((checks.get("session_artifacts") or {}).items())
        )
    )
    print(
        "whatsapp_cloud="
        + ("configured" if cloud.get("cloud_configured") else "not_configured")
        + " missing="
        + (",".join(cloud.get("cloud_missing") or []) or "none")
    )
    print(
        "calling_sidecar="
        + ("configured" if cloud.get("calling_sidecar_configured") else "not_configured")
        + " calling_ready="
        + ("yes" if cloud.get("calling_ready") else "no")
        + " missing="
        + (",".join(cloud.get("calling_missing") or []) or "none")
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
