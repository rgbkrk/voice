#!/usr/bin/env bash
set -euo pipefail

audio_file="${1:?audio file required}"
voice_bin="${VOICE_BIN:-voice}"

exec "$voice_bin" stream-transcribe --quiet "$audio_file"
