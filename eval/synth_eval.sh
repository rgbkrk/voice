#!/bin/bash
# Automated eval using TTS-generated audio (no human recording needed).
# Usage: ./eval/synth_eval.sh [voice_binary]
#
# Generates WAV files from phrases.txt using TTS, then transcribes
# each with the current STT model and compares against expected text.

set -e

VOICE="${1:-./target/release/voice}"
PHRASES="eval/phrases.txt"
TMPDIR="/tmp/voice_synth_eval"

mkdir -p "$TMPDIR"

echo "=== Synthetic STT Evaluation ==="
echo "Using: $VOICE"
echo ""

normalize() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | tr -s ' ' | sed 's/^ //;s/ $//'
}

# Models to test
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
        "$VOICE" say -q -o "$wav" "$phrase"
    fi
    echo "$phrase" > "$TMPDIR/${padded}.txt"
done < "$PHRASES"
echo "Generated $n audio files."
echo ""

for model in "${MODELS[@]}"; do
    echo "=== Model: $model ==="
    pass=0
    fail=0
    start_time=$(date +%s)

    for txt_file in "$TMPDIR"/*.txt; do
        base=$(basename "$txt_file" .txt)
        wav_file="$TMPDIR/${base}.wav"
        [ ! -f "$wav_file" ] && continue

        expected=$(cat "$txt_file")
        transcript=$(STT_MODEL="$model" "$VOICE" transcribe -q "$wav_file" 2>/dev/null || echo "ERROR")

        exp_norm=$(normalize "$expected")
        act_norm=$(normalize "$transcript")

        if [ "$exp_norm" = "$act_norm" ]; then
            printf "  %s PASS\n" "$base"
            pass=$((pass + 1))
        else
            printf "  %s FAIL\n" "$base"
            printf "    expected: %s\n" "$exp_norm"
            printf "    got:      %s\n" "$act_norm"
            fail=$((fail + 1))
        fi
    done

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    total=$((pass + fail))
    echo ""
    echo "  Results: $pass/$total passed ($fail failed) in ${elapsed}s"
    echo ""
done
