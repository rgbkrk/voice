#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

voice_bin="${VOICE_BIN:-voice}"
command_script="${HERMES_TTS_COMMAND_SCRIPT:-$repo_root/examples/hermes-command-tts.sh}"
voice_name="${VOICE_NAME:-${VOICE:-af_heart}}"
speed="${SPEED:-1.0}"
text="${TEXT:-Hermes voice command provider smoke test.}"
keep_output="${KEEP_OUTPUT:-0}"

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required on PATH"
}

require_command ffprobe

if [[ "$voice_bin" == */* ]]; then
  [[ -x "$voice_bin" ]] || fail "VOICE_BIN is not executable: $voice_bin"
else
  require_command "$voice_bin"
fi

[[ -x "$command_script" ]] || fail "Hermes command-provider script is not executable: $command_script"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/voice-hermes-tts.XXXXXX")"
input_path="$tmp_dir/input.txt"
output_path="${OUTPUT:-$tmp_dir/reply.ogg}"

cleanup() {
  if [[ -z "${OUTPUT:-}" && "$keep_output" != "1" ]]; then
    rm -rf "$tmp_dir"
  else
    rm -f "$input_path"
    rmdir "$tmp_dir" 2>/dev/null || true
  fi
}
trap cleanup EXIT

printf '%s\n' "$text" >"$input_path"

VOICE_BIN="$voice_bin" "$command_script" "$input_path" "$output_path" "$voice_name" "$speed"

[[ -s "$output_path" ]] || fail "command provider did not write audio: $output_path"

magic="$(head -c 4 "$output_path" || true)"
[[ "$magic" == "OggS" ]] || fail "expected Ogg container magic, got: ${magic:-<empty>}"

probe="$(
  ffprobe -v error \
    -select_streams a:0 \
    -show_entries stream=codec_name,sample_rate,channels \
    -of default=noprint_wrappers=1 \
    "$output_path"
)"

probe_value() {
  local key="$1"
  printf '%s\n' "$probe" | sed -n "s/^${key}=//p" | head -n 1
}

codec="$(probe_value codec_name)"
sample_rate="$(probe_value sample_rate)"
channels="$(probe_value channels)"

[[ "$codec" == "opus" ]] || fail "expected Opus codec, got: ${codec:-<missing>}"
[[ "$sample_rate" == "48000" ]] || fail "expected 48 kHz sample rate, got: ${sample_rate:-<missing>}"
[[ "$channels" == "1" ]] || fail "expected mono audio, got channels=${channels:-<missing>}"

echo "ok: Hermes command-provider output is Ogg/Opus mono 48 kHz"
echo "voice_bin=$voice_bin"
echo "command_script=$command_script"
if [[ -z "${OUTPUT:-}" && "$keep_output" != "1" ]]; then
  echo "output=<temporary; set KEEP_OUTPUT=1 or OUTPUT=/path/reply.ogg to retain>"
else
  echo "output=$output_path"
fi
