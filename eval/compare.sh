#!/bin/bash
# Compare STT accuracy across models on recorded eval phrases.
# Usage: ./eval/compare.sh [voice_binary]
#
# Transcribes each recording in eval/recordings/ and compares
# against the expected text, computing word error rate.

set -e

VOICE="${1:-./target/release/voice}"
RECORDINGS="eval/recordings"

if [ ! -d "$RECORDINGS" ]; then
    echo "Error: $RECORDINGS not found. Run record.sh first."
    exit 1
fi

# Models to test
MODELS=(
    "distil-whisper/distil-large-v3"
    "distil-whisper/distil-medium.en"
)

echo "=== Voice STT Evaluation ==="
echo "Using: $VOICE"
echo ""

normalize() {
    # Lowercase, strip punctuation, collapse whitespace
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | tr -s ' ' | sed 's/^ //;s/ $//'
}

word_error_rate() {
    local expected="$1"
    local actual="$2"

    # Simple WER: count words that differ
    local exp_words=($expected)
    local act_words=($actual)
    local exp_len=${#exp_words[@]}
    local act_len=${#act_words[@]}
    local max_len=$((exp_len > act_len ? exp_len : act_len))

    if [ "$max_len" -eq 0 ]; then
        echo "0.0"
        return
    fi

    local errors=0
    for i in $(seq 0 $((max_len - 1))); do
        local e="${exp_words[$i]:-}"
        local a="${act_words[$i]:-}"
        if [ "$e" != "$a" ]; then
            errors=$((errors + 1))
        fi
    done

    # WER as percentage
    echo "scale=1; $errors * 100 / $exp_len" | bc
}

for model in "${MODELS[@]}"; do
    echo "=== Model: $model ==="
    total_wer=0
    count=0

    for txt_file in "$RECORDINGS"/*.txt; do
        base=$(basename "$txt_file" .txt)
        wav_file="$RECORDINGS/${base}.wav"

        if [ ! -f "$wav_file" ]; then
            continue
        fi

        expected=$(cat "$txt_file")
        transcript=$(STT_MODEL="$model" "$VOICE" transcribe -q "$wav_file" 2>/dev/null || echo "ERROR")

        exp_norm=$(normalize "$expected")
        act_norm=$(normalize "$transcript")

        wer=$(word_error_rate "$exp_norm" "$act_norm")
        total_wer=$(echo "$total_wer + $wer" | bc)
        count=$((count + 1))

        if [ "$exp_norm" = "$act_norm" ]; then
            status="PASS"
        else
            status="WER=${wer}%"
        fi

        printf "  %s [%s]\n" "$base" "$status"
        if [ "$exp_norm" != "$act_norm" ]; then
            printf "    expected: %s\n" "$exp_norm"
            printf "    got:      %s\n" "$act_norm"
        fi
    done

    if [ "$count" -gt 0 ]; then
        avg_wer=$(echo "scale=1; $total_wer / $count" | bc)
        echo ""
        echo "  Average WER: ${avg_wer}% ($count phrases)"
    fi
    echo ""
done
