#!/usr/bin/env python3
"""Install voice-native Hermes TTS/STT command-provider config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_CONFIG = Path.home() / ".hermes" / "config.yaml"
DEFAULT_TTS_PROVIDER = "kokoro"
DEFAULT_STT_PROVIDER = "voice"
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = "1.0"
DEFAULT_TTS_TIMEOUT = 180
DEFAULT_STT_TIMEOUT = 300
DEFAULT_MAX_TEXT_LENGTH = 2000


class InstallError(Exception):
    """Hermes voice config installation failure."""


def import_yaml():
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise InstallError(
            "PyYAML is required to read/write Hermes config. Install it with "
            "`python -m pip install PyYAML`, or run this script with "
            "`--print-snippet` and copy the YAML manually."
        ) from exc
    return yaml


def resolve_voice_bin(value: str | None) -> str:
    if value:
        return value
    found = shutil.which("voice")
    return found or "voice"


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_tts_command(args: argparse.Namespace, voice_bin: str) -> str:
    if args.mode == "shim":
        shim = args.tts_shim or repo_root() / "examples" / "hermes-command-tts.sh"
        return quote_command(
            [
                str(shim),
                "{input_path}",
                "{output_path}",
                "{voice}",
                "{speed}",
            ]
        )

    return quote_command(
        [
            voice_bin,
            "say",
            "--format",
            "ogg-opus",
            "--input-file",
            "{input_path}",
            "--output",
            "{output_path}",
            "--voice",
            "{voice}",
            "--speed",
            "{speed}",
        ]
    )


def build_stt_command(args: argparse.Namespace, voice_bin: str) -> str:
    if args.mode == "shim":
        shim = args.stt_shim or repo_root() / "examples" / "hermes-command-stt.sh"
        return quote_command([str(shim), "{input_path}"])
    return quote_command([voice_bin, "stream-transcribe", "--quiet", "{input_path}"])


def provider_blocks(args: argparse.Namespace, voice_bin: str) -> dict[str, Any]:
    return {
        "tts": {
            "provider": args.tts_provider,
            "providers": {
                args.tts_provider: {
                    "type": "command",
                    "command": build_tts_command(args, voice_bin),
                    "output_format": "ogg",
                    "voice_compatible": True,
                    "voice": args.voice,
                    "speed": str(args.speed),
                    "timeout": int(args.tts_timeout),
                    "max_text_length": int(args.max_text_length),
                }
            },
        },
        "stt": {
            "enabled": True,
            "provider": args.stt_provider,
            "providers": {
                args.stt_provider: {
                    "type": "command",
                    "command": build_stt_command(args, voice_bin),
                    "format": "txt",
                    "timeout": int(args.stt_timeout),
                }
            },
        },
    }


def render_snippet(blocks: dict[str, Any]) -> str:
    lines: list[str] = []
    tts_provider = next(iter(blocks["tts"]["providers"]))
    tts = blocks["tts"]["providers"][tts_provider]
    stt_provider = next(iter(blocks["stt"]["providers"]))
    stt = blocks["stt"]["providers"][stt_provider]
    lines.extend(
        [
            "tts:",
            f"  provider: {blocks['tts']['provider']}",
            "  providers:",
            f"    {tts_provider}:",
            "      type: command",
            f"      command: {tts['command']}",
            "      output_format: ogg",
            "      voice_compatible: true",
            f"      voice: {tts['voice']}",
            f"      speed: {tts['speed']}",
            f"      timeout: {tts['timeout']}",
            f"      max_text_length: {tts['max_text_length']}",
            "",
            "stt:",
            "  enabled: true",
            f"  provider: {blocks['stt']['provider']}",
            "  providers:",
            f"    {stt_provider}:",
            "      type: command",
            f"      command: {stt['command']}",
            "      format: txt",
            f"      timeout: {stt['timeout']}",
        ]
    )
    return "\n".join(lines) + "\n"


def load_config(path: Path, *, create: bool) -> dict[str, Any]:
    yaml = import_yaml()
    if not path.exists():
        if create:
            return {}
        raise InstallError(f"Hermes config not found: {path}; pass --create to create it")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InstallError(f"failed to parse YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise InstallError(f"{path} must contain a YAML mapping")
    return loaded


def ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    if not isinstance(value, dict):
        raise InstallError(f"config key {key!r} must be a mapping")
    return value


def merge_provider_config(config: dict[str, Any], blocks: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)

    tts = ensure_mapping(updated, "tts")
    tts["provider"] = blocks["tts"]["provider"]
    tts_providers = ensure_mapping(tts, "providers")
    for provider, provider_config in blocks["tts"]["providers"].items():
        existing = tts_providers.get(provider)
        merged = existing.copy() if isinstance(existing, dict) else {}
        merged.update(provider_config)
        tts_providers[provider] = merged

    stt = ensure_mapping(updated, "stt")
    stt["enabled"] = True
    stt["provider"] = blocks["stt"]["provider"]
    stt_providers = ensure_mapping(stt, "providers")
    for provider, provider_config in blocks["stt"]["providers"].items():
        existing = stt_providers.get(provider)
        merged = existing.copy() if isinstance(existing, dict) else {}
        merged.update(provider_config)
        stt_providers[provider] = merged

    return updated


def write_yaml_atomic(path: Path, config: dict[str, Any]) -> None:
    yaml = import_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def backup_config(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    backup = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup)
    return backup


def run_verifier(path: Path, voice_bin: str, *, run_tts_smoke: bool) -> str:
    verifier = Path(__file__).resolve().with_name("verify_hermes_voice_config.py")
    command = [
        sys.executable,
        str(verifier),
        "--config",
        str(path),
        "--voice-bin",
        voice_bin,
    ]
    if not run_tts_smoke:
        command.append("--skip-tts-smoke")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise InstallError(f"post-install verifier failed: {detail}")
    return completed.stdout


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install voice-native Hermes TTS/STT command-provider config.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--voice-bin", default=None)
    parser.add_argument(
        "--mode",
        choices=["direct", "shim"],
        default="direct",
        help="install direct voice commands or repo-owned command shims",
    )
    parser.add_argument("--tts-provider", default=DEFAULT_TTS_PROVIDER)
    parser.add_argument("--stt-provider", default=DEFAULT_STT_PROVIDER)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--speed", default=DEFAULT_SPEED)
    parser.add_argument("--tts-timeout", type=int, default=DEFAULT_TTS_TIMEOUT)
    parser.add_argument("--stt-timeout", type=int, default=DEFAULT_STT_TIMEOUT)
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    parser.add_argument("--tts-shim", type=Path, default=None)
    parser.add_argument("--stt-shim", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="write the config")
    parser.add_argument("--create", action="store_true", help="create the config if missing")
    parser.add_argument("--no-backup", action="store_true", help="do not keep a .bak copy")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="do not run verify_hermes_voice_config.py on the resulting config",
    )
    parser.add_argument(
        "--run-tts-smoke",
        action="store_true",
        help="let post-install verification execute the configured TTS command",
    )
    parser.add_argument(
        "--print-snippet",
        action="store_true",
        help="print only the YAML snippet; does not require PyYAML",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    voice_bin = resolve_voice_bin(args.voice_bin)
    blocks = provider_blocks(args, voice_bin)

    if args.print_snippet:
        print(render_snippet(blocks), end="")
        return 0

    config_path = args.config.expanduser()
    config = load_config(config_path, create=args.create)
    updated = merge_provider_config(config, blocks)

    if not args.apply:
        print("dry_run=true")
        print(f"config={config_path}")
        print("applied=false")
        print(f"mode={args.mode}")
        print(f"tts.provider={args.tts_provider}")
        print(f"stt.provider={args.stt_provider}")
        print("snippet:")
        print(render_snippet(blocks), end="")
        if not args.skip_verify:
            with tempfile.TemporaryDirectory(prefix="voice-hermes-install.") as tmp:
                temp_config = Path(tmp) / "config.yaml"
                write_yaml_atomic(temp_config, updated)
                run_verifier(temp_config, voice_bin, run_tts_smoke=args.run_tts_smoke)
            print("verify=passed")
        return 0

    backup = None
    if not args.no_backup:
        backup = backup_config(config_path)
    write_yaml_atomic(config_path, updated)
    if not args.skip_verify:
        run_verifier(config_path, voice_bin, run_tts_smoke=args.run_tts_smoke)

    print("ok: Hermes voice config installed")
    print(f"config={config_path}")
    print("applied=true")
    print(f"mode={args.mode}")
    print(f"tts.provider={args.tts_provider}")
    print(f"stt.provider={args.stt_provider}")
    print(f"backup={backup or 'none'}")
    print(f"verify={'skipped' if args.skip_verify else 'passed'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (InstallError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
