#!/usr/bin/env python3
"""Verify that a local Hermes config uses voice-native TTS/STT surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_CONFIG = Path.home() / ".hermes" / "config.yaml"


class ConfigError(Exception):
    """Configuration validation failure."""


def minimal_yaml_load(text: str) -> dict[str, Any]:
    """Load the small YAML subset needed for Hermes config checks.

    PyYAML is preferred when available. This fallback handles nested maps,
    booleans, numbers, quoted scalars, and folded plain scalars such as the
    wrapped command strings Hermes writes in config.yaml.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    last_scalar: tuple[int, dict[str, Any], str] | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if line.startswith("- "):
            last_scalar = None
            continue

        if ":" not in line:
            if last_scalar is not None and indent > last_scalar[0]:
                _indent, parent, key = last_scalar
                value = parent.get(key)
                if isinstance(value, str):
                    parent[key] = f"{value} {line}"
            continue

        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            stack = [(-1, root)]

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        parent = stack[-1][1]

        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            last_scalar = None
        else:
            parent[key] = parse_scalar(value)
            last_scalar = (indent, parent, key)

    return root


def parse_scalar(value: str) -> Any:
    if value in {"''", '""'}:
        return ""
    if (
        (value.startswith("'") and value.endswith("'"))
        or (value.startswith('"') and value.endswith('"'))
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        loaded = minimal_yaml_load(text)
    else:
        try:
            loaded = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigError(f"failed to parse YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} did not contain a YAML mapping")
    return loaded


def get_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def lookup(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"missing config key: {dotted_path}")
        value = value[part]
    return value


def as_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ConfigError(f"{path} must be a boolean")


def as_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value.strip()


def split_command(command: str) -> list[str]:
    try:
        args = shlex.split(command)
    except ValueError as exc:
        raise ConfigError(f"command is not shell-parseable: {exc}") from exc
    if not args:
        raise ConfigError("command must not be empty")
    return args


def option_value(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name and index + 1 < len(args):
            return args[index + 1]
    return None


def require_placeholder(args: list[str], option: str, placeholder: str) -> None:
    value = option_value(args, option)
    if value != placeholder:
        raise ConfigError(
            f"tts command must pass {option} {placeholder}; got {value or '<missing>'}"
        )


def is_voice_command(args: list[str]) -> bool:
    return Path(args[0]).name == "voice"


def validate_tts(config: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    provider = as_string(lookup(config, "tts.provider"), "tts.provider")
    providers = get_mapping(lookup(config, "tts.providers"), "tts.providers")
    provider_config = get_mapping(
        providers.get(provider), f"tts.providers.{provider}"
    )

    provider_type = as_string(provider_config.get("type"), f"tts.providers.{provider}.type")
    if provider_type != "command":
        raise ConfigError(f"tts provider {provider!r} must have type: command")

    output_format = as_string(
        provider_config.get("output_format"),
        f"tts.providers.{provider}.output_format",
    )
    if output_format != "ogg":
        raise ConfigError(
            f"tts provider {provider!r} must use output_format: ogg; got {output_format!r}"
        )

    voice_compatible = as_bool(
        provider_config.get("voice_compatible"),
        f"tts.providers.{provider}.voice_compatible",
    )
    if not voice_compatible:
        raise ConfigError(f"tts provider {provider!r} must set voice_compatible: true")

    command = as_string(provider_config.get("command"), f"tts.providers.{provider}.command")
    args = split_command(command)
    if not is_voice_command(args):
        raise ConfigError("tts command must invoke the voice binary directly")
    if "say" not in args[1:]:
        raise ConfigError("tts command must invoke `voice say`")
    if option_value(args, "--format") != "ogg-opus":
        raise ConfigError("tts command must pass `--format ogg-opus`")
    require_placeholder(args, "--input-file", "{input_path}")
    require_placeholder(args, "--output", "{output_path}")
    require_placeholder(args, "--voice", "{voice}")
    require_placeholder(args, "--speed", "{speed}")
    return provider, provider_config, args


def validate_stt(config: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    enabled = as_bool(lookup(config, "stt.enabled"), "stt.enabled")
    if not enabled:
        raise ConfigError("stt.enabled must be true")

    provider = as_string(lookup(config, "stt.provider"), "stt.provider")
    providers = get_mapping(lookup(config, "stt.providers"), "stt.providers")
    provider_config = get_mapping(
        providers.get(provider), f"stt.providers.{provider}"
    )

    provider_type = as_string(provider_config.get("type"), f"stt.providers.{provider}.type")
    if provider_type != "command":
        raise ConfigError(f"stt provider {provider!r} must have type: command")

    command = as_string(provider_config.get("command"), f"stt.providers.{provider}.command")
    args = split_command(command)
    if not is_voice_command(args):
        raise ConfigError("stt command must invoke the voice binary directly")
    if "stream-transcribe" not in args[1:]:
        raise ConfigError("stt command must invoke `voice stream-transcribe`")
    if "{input_path}" not in args:
        raise ConfigError("stt command must pass {input_path}")
    if "--quiet" not in args:
        raise ConfigError("stt command must pass --quiet so Hermes receives transcript text")

    output_format = as_string(provider_config.get("format"), f"stt.providers.{provider}.format")
    if output_format != "txt":
        raise ConfigError(f"stt provider {provider!r} must use format: txt")
    return provider, provider_config, args


def substitute_command(
    args: list[str],
    *,
    input_path: Path,
    output_path: Path,
    voice: str,
    speed: str,
    voice_bin: str | None,
) -> list[str]:
    resolved = []
    for index, arg in enumerate(args):
        if index == 0 and voice_bin is not None:
            resolved.append(voice_bin)
            continue
        resolved.append(
            arg.replace("{input_path}", str(input_path))
            .replace("{output_path}", str(output_path))
            .replace("{voice}", voice)
            .replace("{speed}", speed)
        )
    return resolved


def probe_ogg_opus(path: Path) -> dict[str, str]:
    if shutil.which("ffprobe") is None:
        raise ConfigError("ffprobe is required for --run-tts-smoke")
    if not path.is_file() or path.stat().st_size == 0:
        raise ConfigError(f"tts smoke did not write audio: {path}")
    if path.read_bytes()[:4] != b"OggS":
        raise ConfigError(f"tts smoke output is not an Ogg container: {path}")

    result = subprocess.run(
        [
            "ffprobe",
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
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ConfigError("ffprobe did not find an audio stream")
    stream = streams[0]
    codec = str(stream.get("codec_name") or "")
    sample_rate = str(stream.get("sample_rate") or "")
    channels = str(stream.get("channels") or "")
    if codec != "opus":
        raise ConfigError(f"expected Opus codec, got {codec or '<missing>'}")
    if sample_rate != "48000":
        raise ConfigError(f"expected 48 kHz sample rate, got {sample_rate or '<missing>'}")
    if channels != "1":
        raise ConfigError(f"expected mono output, got channels={channels or '<missing>'}")
    return {"codec": codec, "sample_rate": sample_rate, "channels": channels}


def run_tts_smoke(
    provider_config: dict[str, Any],
    command_args: list[str],
    *,
    voice_bin: str | None,
    text: str,
) -> dict[str, str]:
    voice = str(provider_config.get("voice") or "af_heart")
    speed = str(provider_config.get("speed") or "1.0")
    timeout = float(provider_config.get("timeout") or 180)

    with tempfile.TemporaryDirectory(prefix="voice-hermes-config.") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.txt"
        output_path = tmp_path / "reply.ogg"
        input_path.write_text(text, encoding="utf-8")

        args = substitute_command(
            command_args,
            input_path=input_path,
            output_path=output_path,
            voice=voice,
            speed=speed,
            voice_bin=voice_bin,
        )
        subprocess.run(args, check=True, timeout=timeout)
        return probe_ogg_opus(output_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that ~/.hermes/config.yaml uses voice-native TTS/STT.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--voice-bin",
        help="override the first token of direct voice commands for smoke execution",
    )
    parser.add_argument(
        "--skip-tts-smoke",
        action="store_true",
        help="only validate config shape; do not execute the configured TTS command",
    )
    parser.add_argument(
        "--skip-stt-config",
        action="store_true",
        help="skip STT command-provider validation",
    )
    parser.add_argument(
        "--text",
        default="Hermes voice-native configuration smoke test.",
        help="text to synthesize when running the TTS smoke",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser()
    if not config_path.is_file():
        raise ConfigError(f"Hermes config not found: {config_path}")

    config = load_yaml(config_path)
    tts_provider, tts_config, tts_args = validate_tts(config)
    stt_provider = None
    if not args.skip_stt_config:
        stt_provider, _stt_config, _stt_args = validate_stt(config)

    smoke = "skipped"
    probe = None
    if not args.skip_tts_smoke:
        probe = run_tts_smoke(
            tts_config,
            tts_args,
            voice_bin=args.voice_bin,
            text=args.text,
        )
        smoke = "checked"

    print("ok: Hermes voice config verifier passed")
    print(f"config={config_path}")
    print(f"tts.provider={tts_provider}")
    print("tts.output_format=ogg")
    print("tts.voice_compatible=true")
    print("tts.command=voice say --format ogg-opus")
    print(f"tts_smoke={smoke}")
    if probe is not None:
        print(
            "tts_probe="
            f"codec={probe['codec']},sample_rate={probe['sample_rate']},channels={probe['channels']}"
        )
    if stt_provider is not None:
        print(f"stt.provider={stt_provider}")
        print("stt.command=voice stream-transcribe --quiet")
    else:
        print("stt_config=skipped")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ConfigError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
