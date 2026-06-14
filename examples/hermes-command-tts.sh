#!/usr/bin/env bash
set -euo pipefail

text_file="${1:?text file required}"
out_file="${2:?output file required}"
voice="${3:-af_heart}"
speed="${4:-1.0}"
voice_bin="${VOICE_BIN:-voice}"
format="${VOICE_FORMAT-ogg-opus}"

args=(
  say
  --input-file "$text_file"
  --output "$out_file"
  --voice "$voice"
  --speed "$speed"
)

if [[ -n "$format" ]]; then
  args+=(--format "$format")
fi

exec "$voice_bin" "${args[@]}"
