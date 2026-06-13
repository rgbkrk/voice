#!/usr/bin/env bash
set -euo pipefail

text_file="${1:?text file required}"
out_file="${2:?output file required}"
voice="${3:-af_heart}"
speed="${4:-1.0}"
voice_bin="${VOICE_BIN:-voice}"

exec "$voice_bin" say \
  --input-file "$text_file" \
  --output "$out_file" \
  --voice "$voice" \
  --speed "$speed"
