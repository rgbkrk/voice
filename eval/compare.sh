#!/bin/bash
# Compare STT accuracy across models on recorded eval phrases.
# Usage: ./eval/compare.sh [voice_binary]
#
# Transcribes each recording in eval/recordings/ and compares
# against the expected text, computing WER/CER and latency metrics.

set -euo pipefail

VOICE="${1:-./target/release/voice}"
RECORDINGS="eval/recordings"
RESULTS_DIR="eval/results"

if [ ! -d "$RECORDINGS" ]; then
    echo "Error: $RECORDINGS not found. Run record.sh first."
    exit 1
fi

# Rust voice STT models to test
MODELS=(
    "distil-whisper/distil-large-v3"
    "distil-whisper/distil-medium.en"
)

echo "=== Voice STT Evaluation ==="
echo "Using: $VOICE"
echo ""

mkdir -p "$RESULTS_DIR"

for model in "${MODELS[@]}"; do
    safe_model="${model//\//_}"
    python3 eval/evaluate.py \
        --recordings "$RECORDINGS" \
        --voice "$VOICE" \
        --model "$model" \
        --json-out "$RESULTS_DIR/${safe_model}.json"
    echo ""
done

if [ "${PARAKEET_MLX:-0}" = "1" ]; then
    model="${PARAKEET_MODEL:-mlx-community/parakeet-tdt-0.6b-v3}"
    safe_model="${model//\//_}"
    uv run --with parakeet-mlx python eval/evaluate.py \
        --recordings "$RECORDINGS" \
        --engine parakeet-mlx \
        --model "$model" \
        --json-out "$RESULTS_DIR/parakeet_mlx_${safe_model}.json"
    echo ""
fi
