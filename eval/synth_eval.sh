#!/bin/bash
# Automated eval using TTS-generated audio (no human recording needed).
# Usage: ./eval/synth_eval.sh [voice_binary]
#
# Generates deterministic WAV files from phrases.txt using TTS, then
# transcribes each with the current STT model and compares against expected text.

set -euo pipefail

VOICE="${1:-./target/release/voice}"
PHRASES="eval/phrases.txt"
TMPDIR="/tmp/voice_synth_eval_deterministic"
RESULTS_DIR="eval/results"

mkdir -p "$TMPDIR"
mkdir -p "$RESULTS_DIR"

echo "=== Synthetic STT Evaluation ==="
echo "Using: $VOICE"
echo ""

# Rust voice STT models to test
MODELS=(
    "distil-whisper/distil-large-v3"
    "distil-whisper/distil-medium.en"
)

# Generate TTS audio first
echo "Generating TTS audio..."
n=0
while IFS= read -r phrase; do
    [ -z "$phrase" ] && continue
    n=$((n + 1))
    padded=$(printf "%03d" "$n")
    wav="$TMPDIR/${padded}.wav"
    if [ ! -f "$wav" ]; then
        "$VOICE" say --deterministic -q -o "$wav" "$phrase"
    fi
    echo "$phrase" > "$TMPDIR/${padded}.txt"
done < "$PHRASES"
echo "Generated $n audio files."
echo ""

for model in "${MODELS[@]}"; do
    safe_model="${model//\//_}"
    python3 eval/evaluate.py \
        --recordings "$TMPDIR" \
        --voice "$VOICE" \
        --model "$model" \
        --json-out "$RESULTS_DIR/synth_${safe_model}.json"
    echo ""
done

if [ "${PARAKEET_MLX:-0}" = "1" ]; then
    model="${PARAKEET_MODEL:-mlx-community/parakeet-tdt-0.6b-v3}"
    safe_model="${model//\//_}"
    uv run --with parakeet-mlx python eval/evaluate.py \
        --recordings "$TMPDIR" \
        --engine parakeet-mlx \
        --model "$model" \
        --json-out "$RESULTS_DIR/synth_parakeet_mlx_${safe_model}.json"
    echo ""
fi
