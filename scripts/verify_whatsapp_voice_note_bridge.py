#!/usr/bin/env python3
"""Generate a WhatsApp voice note with voice and optionally send it via Baileys."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_whatsapp_bridge_runtime import (  # noqa: E402
    DEFAULT_BRIDGE_URL,
    fetch_bridge_health,
    find_bridge_processes,
    parse_env_file,
)


DEFAULT_TEXT = "Voice runtime smoke test from voice."


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
    if found:
        return found
    return "voice"


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


def generate_voice_note(
    voice_bin: str,
    *,
    output_path: Path,
    text: str,
    voice: str,
    speed: str,
    timeout: float,
) -> dict[str, Any]:
    command = [
        voice_bin,
        "say",
        "--quiet",
        "--format",
        "ogg-opus",
        "--output",
        str(output_path),
        "--voice",
        voice,
        "--speed",
        speed,
        text,
    ]
    completed = run_command(command, timeout=timeout)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "stdout": completed.stdout.strip(),
    }


def probe_ogg(path: Path, *, timeout: float, skip_ffprobe: bool) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    probe: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "magic": None,
        "ffprobe": None,
    }
    if not path.is_file():
        return probe, [f"voice note file was not created: {path}"]

    with path.open("rb") as handle:
        probe["magic"] = handle.read(4).decode("ascii", errors="replace")
    if probe["magic"] != "OggS":
        failures.append(f"voice note file is not an Ogg container: magic={probe['magic']!r}")
    if probe["size_bytes"] <= 64:
        failures.append(f"voice note file is too small: {probe['size_bytes']} bytes")

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
            "stream=codec_name,sample_rate,channels",
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
        failures.append(f"ffprobe failed: {completed.stderr.strip()}")
        return probe, failures

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        failures.append(f"ffprobe did not return JSON: {exc}")
        return probe, failures

    stream = (payload.get("streams") or [{}])[0]
    probe["ffprobe"]["stream"] = stream
    if stream.get("codec_name") != "opus":
        failures.append(f"voice note codec={stream.get('codec_name')!r}, expected 'opus'")
    if str(stream.get("sample_rate")) != "48000":
        failures.append(f"voice note sample_rate={stream.get('sample_rate')!r}, expected 48000")
    if int(stream.get("channels") or 0) != 1:
        failures.append(f"voice note channels={stream.get('channels')!r}, expected 1")
    return probe, failures


def infer_media_payload_js(explicit: Path | None, *, bridge_url: str) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = find_bridge_processes(port=port_from_url(bridge_url))
    for process in candidates:
        script = process.get("script")
        if not script:
            continue
        media_payload = Path(str(script)).resolve().parent / "media-payload.js"
        if media_payload.is_file():
            return media_payload
    return None


def port_from_url(url: str) -> int | None:
    pieces = url.rstrip("/").rsplit(":", 1)
    if len(pieces) != 2:
        return None
    try:
        return int(pieces[1].split("/", 1)[0])
    except ValueError:
        return None


def verify_ptt_payload(
    voice_note_path: Path,
    *,
    media_payload_js: Path | None,
    timeout: float,
    skip_ptt_payload_check: bool,
) -> tuple[dict[str, Any], list[str]]:
    if skip_ptt_payload_check:
        return {"skipped": True, "reason": "skipped by flag"}, []
    if media_payload_js is None or not media_payload_js.is_file():
        return {"skipped": True, "reason": "media-payload.js not found"}, []

    node = shutil.which("node")
    if not node:
        return {"skipped": True, "reason": "node not found"}, []

    code = f"""
        import {{ buildMediaPayload }} from {json.dumps(media_payload_js.as_uri())};
        const result = buildMediaPayload({{ filePath: process.env.VOICE_NOTE_FILE, mediaType: 'audio' }});
        const payload = result.payload || {{}};
        console.log(JSON.stringify({{
          hasAudio: Boolean(payload.audio),
          mimetype: payload.mimetype || null,
          ptt: payload.ptt === true,
          warning: result.warning || null
        }}));
    """
    env = {**os.environ, "VOICE_NOTE_FILE": str(voice_note_path)}
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    check: dict[str, Any] = {
        "media_payload_js": str(media_payload_js),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        return check, [f"bridge media payload check failed: {completed.stderr.strip()}"]
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return check, [f"bridge media payload check did not return JSON: {exc}"]
    check.update(payload)

    failures: list[str] = []
    if not check.get("hasAudio"):
        failures.append("bridge media payload did not create an audio payload")
    if check.get("mimetype") != "audio/ogg; codecs=opus":
        failures.append(
            "bridge media payload mimetype="
            f"{check.get('mimetype')!r}, expected 'audio/ogg; codecs=opus'"
        )
    if check.get("ptt") is not True:
        failures.append("bridge media payload did not set ptt=true")
    if check.get("warning"):
        failures.append(f"bridge media payload emitted warning: {check['warning']}")
    return check, failures


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc

    try:
        result = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"POST {url} did not return JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"POST {url} JSON response must be an object")
    return result


def get_json(url: str, *, timeout: float) -> Any:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc
    return json.loads(response_body.decode("utf-8"))


def poll_inbound_audio(
    bridge_url: str,
    *,
    wait_seconds: float,
    timeout: float,
    get_json_func: Callable[..., Any] = get_json,
) -> dict[str, Any]:
    message_endpoint = bridge_url.rstrip("/") + "/messages"
    deadline = time.monotonic() + wait_seconds
    seen: list[dict[str, Any]] = []
    audio_events: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        payload = get_json_func(message_endpoint, timeout=timeout)
        if isinstance(payload, list):
            for event in payload:
                if not isinstance(event, dict):
                    continue
                seen.append(
                    {
                        "chatId": event.get("chatId"),
                        "senderId": event.get("senderId"),
                        "hasMedia": event.get("hasMedia"),
                        "mediaType": event.get("mediaType"),
                        "mediaUrls": event.get("mediaUrls"),
                    }
                )
                if event.get("hasMedia") and event.get("mediaType") in {"audio", "ptt"}:
                    audio_events.append(seen[-1])
        if audio_events:
            break
        time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))
    return {
        "wait_seconds": wait_seconds,
        "message_endpoint": message_endpoint,
        "drains_bridge_messages": True,
        "seen_events": seen,
        "audio_events": audio_events,
    }


def resolve_chat_id(args: argparse.Namespace, env: dict[str, str]) -> str | None:
    if args.chat_id:
        return args.chat_id
    return env.get("WHATSAPP_HOME_CHANNEL") or os.environ.get("WHATSAPP_HOME_CHANNEL")


def verify(
    args: argparse.Namespace,
    *,
    post_json_func: Callable[..., dict[str, Any]] = post_json,
    get_json_func: Callable[..., Any] = get_json,
) -> dict[str, Any]:
    hermes_home = args.hermes_home.expanduser().resolve()
    env_file = args.env_file.expanduser().resolve() if args.env_file else hermes_home / ".env"
    env = parse_env_file(env_file)
    bridge_url = args.bridge_url.rstrip("/")
    voice_bin = resolve_executable(args.voice_bin, label="voice binary")

    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {
        "voice_bin": voice_bin,
        "bridge_url": bridge_url,
        "hermes_home": str(hermes_home),
        "env_file": str(env_file),
        "send_requested": bool(args.send),
        "chat_id": resolve_chat_id(args, env),
    }

    inbound_requested = args.wait_inbound_seconds > 0
    if inbound_requested and not args.drain_bridge_messages:
        failures.append(
            "inbound WhatsApp receive polling drains the bridge /messages queue; "
            "pass --drain-bridge-messages only during an attended receive test"
        )

    if not args.skip_bridge_health:
        health, health_error = fetch_bridge_health(bridge_url, timeout=args.timeout)
        checks["bridge_health"] = health or {}
        if health_error:
            failures.append(health_error)
        elif health and health.get("status") != "connected":
            failures.append(
                f"WhatsApp bridge status={health.get('status')!r}, expected 'connected'"
            )
    else:
        checks["bridge_health"] = {"skipped": True}

    with tempfile.TemporaryDirectory(prefix="voice-whatsapp-note-") as tmp:
        tmp_path = Path(tmp)
        output_path = args.output.expanduser().resolve() if args.output else tmp_path / "voice-note.ogg"
        generate = generate_voice_note(
            voice_bin,
            output_path=output_path,
            text=args.text,
            voice=args.voice,
            speed=args.speed,
            timeout=args.voice_timeout,
        )
        checks["generate"] = generate
        if generate["returncode"] != 0:
            failures.append(f"voice say failed: {generate['stderr']}")

        if output_path.exists():
            probe, probe_failures = probe_ogg(
                output_path,
                timeout=args.timeout,
                skip_ffprobe=args.skip_ffprobe,
            )
            checks["voice_note"] = probe
            failures.extend(probe_failures)

            media_payload_js = infer_media_payload_js(
                args.bridge_media_payload_js,
                bridge_url=bridge_url,
            )
            ptt_check, ptt_failures = verify_ptt_payload(
                output_path,
                media_payload_js=media_payload_js,
                timeout=args.timeout,
                skip_ptt_payload_check=args.skip_ptt_payload_check,
            )
            checks["bridge_ptt_payload"] = ptt_check
            failures.extend(ptt_failures)
            if ptt_check.get("skipped"):
                warnings.append(f"bridge ptt payload check skipped: {ptt_check.get('reason')}")
        else:
            checks["voice_note"] = {"path": str(output_path), "exists": False}

        if args.send:
            chat_id = checks["chat_id"]
            if not chat_id:
                failures.append(
                    "no chat id configured; pass --chat-id or set WHATSAPP_HOME_CHANNEL"
                )
            elif output_path.exists() and not failures:
                payload = {
                    "chatId": chat_id,
                    "filePath": str(output_path),
                    "mediaType": "audio",
                }
                try:
                    send_result = post_json_func(
                        bridge_url + "/send-media",
                        payload,
                        timeout=args.timeout,
                    )
                    checks["send_media"] = {
                        "request": {
                            "chatId": chat_id,
                            "filePath": str(output_path),
                            "mediaType": "audio",
                        },
                        "response": send_result,
                    }
                    if send_result.get("success") is not True:
                        failures.append(f"bridge send-media did not report success: {send_result}")
                except Exception as exc:
                    failures.append(str(exc))
            else:
                checks["send_media"] = {"skipped": True, "reason": "prior failures"}
        else:
            checks["send_media"] = {"skipped": True, "reason": "pass --send to post"}

        if args.wait_inbound_seconds > 0:
            inbound_base = {
                "wait_seconds": args.wait_inbound_seconds,
                "message_endpoint": bridge_url + "/messages",
                "drains_bridge_messages": bool(args.drain_bridge_messages),
            }
            if not args.drain_bridge_messages:
                checks["inbound_audio"] = {
                    **inbound_base,
                    "skipped": True,
                    "reason": "requires --drain-bridge-messages",
                }
            elif failures:
                checks["inbound_audio"] = {
                    **inbound_base,
                    "skipped": True,
                    "reason": "prior failures",
                }
            else:
                try:
                    inbound = poll_inbound_audio(
                        bridge_url,
                        wait_seconds=args.wait_inbound_seconds,
                        timeout=args.timeout,
                        get_json_func=get_json_func,
                    )
                    checks["inbound_audio"] = inbound
                    if args.require_inbound_audio and not inbound["audio_events"]:
                        failures.append(
                            "no inbound WhatsApp audio event arrived during wait window"
                        )
                except Exception as exc:
                    failures.append(str(exc))
        else:
            checks["inbound_audio"] = {
                "skipped": True,
                "drains_bridge_messages": False,
            }

        if args.keep_output and output_path.exists():
            checks["retained_output"] = str(output_path)
        elif args.output and output_path.exists():
            checks["retained_output"] = str(output_path)
        else:
            checks["retained_output"] = None

        if args.keep_output and not args.output and output_path.exists():
            retained = Path.cwd() / f"voice-whatsapp-note-{int(time.time())}.ogg"
            shutil.copy2(output_path, retained)
            checks["retained_output"] = str(retained)

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
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--bridge-url", default=os.environ.get("WHATSAPP_BRIDGE_URL", DEFAULT_BRIDGE_URL))
    parser.add_argument("--chat-id", default=None)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bridge-media-payload-js", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--voice-timeout", type=float, default=180.0)
    parser.add_argument("--skip-bridge-health", action="store_true")
    parser.add_argument("--skip-ffprobe", action="store_true")
    parser.add_argument("--skip-ptt-payload-check", action="store_true")
    parser.add_argument("--send", action="store_true", help="post a real WhatsApp voice note")
    parser.add_argument("--wait-inbound-seconds", type=float, default=0.0)
    parser.add_argument("--require-inbound-audio", action="store_true")
    parser.add_argument(
        "--drain-bridge-messages",
        action="store_true",
        help=(
            "allow attended inbound receive polling via GET /messages; this consumes "
            "queued bridge messages"
        ),
    )
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def human_summary(result: dict[str, Any]) -> None:
    checks = result["checks"]
    if result["success"]:
        print("ok: WhatsApp voice-note bridge smoke passed")
    else:
        print("error: WhatsApp voice-note bridge smoke failed", file=sys.stderr)
        for failure in result["failures"]:
            print(f"- {failure}", file=sys.stderr)

    note = checks.get("voice_note") or {}
    ptt = checks.get("bridge_ptt_payload") or {}
    send = checks.get("send_media") or {}
    print(f"voice_bin={checks.get('voice_bin')}")
    print(f"bridge={checks.get('bridge_url')}")
    print(
        "voice_note="
        f"path={note.get('path')} size={note.get('size_bytes')} magic={note.get('magic')}"
    )
    if ptt.get("skipped"):
        print(f"bridge_ptt_payload=skipped reason={ptt.get('reason')}")
    else:
        print(
            "bridge_ptt_payload="
            f"ptt={ptt.get('ptt')} mimetype={ptt.get('mimetype')} "
            f"has_audio={ptt.get('hasAudio')}"
        )
    if send.get("skipped"):
        print(f"send_media=skipped reason={send.get('reason')}")
    else:
        response = send.get("response") or {}
        print(
            "send_media="
            f"success={response.get('success')} message_id={response.get('messageId')}"
        )
    inbound = checks.get("inbound_audio") or {}
    if inbound.get("skipped"):
        reason = inbound.get("reason")
        if reason:
            print(
                "inbound_audio="
                f"skipped reason={reason} "
                f"drains_messages={inbound.get('drains_bridge_messages')}"
            )
    else:
        print(
            "inbound_audio="
            f"events={len(inbound.get('seen_events') or [])} "
            f"audio_events={len(inbound.get('audio_events') or [])} "
            f"drains_messages={inbound.get('drains_bridge_messages')}"
        )
    if checks.get("retained_output"):
        print(f"retained_output={checks['retained_output']}")
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
