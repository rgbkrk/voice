#!/bin/bash
# Record eval phrases one at a time.
# Usage: ./eval/record.sh [voice_binary]
#
# For each phrase in phrases.txt:
#   1. Speaks the phrase aloud (so you know what to say)
#   2. Records your voice reading it back (auto-stops on silence)
#   3. Saves the recording as eval/recordings/NNN.wav
#   4. Saves the expected text as eval/recordings/NNN.txt

set -e

VOICE="${1:-./target/release/voice}"
PHRASES="eval/phrases.txt"
OUTDIR="eval/recordings"

mkdir -p "$OUTDIR"

if [ ! -f "$PHRASES" ]; then
    echo "Error: $PHRASES not found"
    exit 1
fi

echo "=== Voice STT Eval Recording Session ==="
echo "Using: $VOICE"
echo ""
echo "For each phrase:"
echo "  1. I'll say the phrase so you know what to read"
echo "  2. After the ding, read the exact phrase back"
echo "  3. Recording auto-stops after you go silent"
echo ""
echo "Press Enter to start..."
read -r

n=0
while IFS= read -r phrase; do
    [ -z "$phrase" ] && continue
    n=$((n + 1))
    padded=$(printf "%03d" "$n")

    echo ""
    echo "--- Phrase $n/$(wc -l < "$PHRASES" | tr -d ' ') ---"
    echo "  \"$phrase\""
    echo ""

    # Save expected text
    echo "$phrase" > "$OUTDIR/${padded}.txt"

    # Speak the phrase so the user knows what to say
    "$VOICE" say -q "$phrase"

    echo "  Your turn — read the phrase after the ding..."

    # Record with VOICE_SAVE_RECORDING=1 to get the WAV file
    export VOICE_SAVE_RECORDING=1
    transcript=$(VOICE_SAVE_RECORDING=1 "$VOICE" listen 2>/tmp/voice_eval_stderr.txt)

    echo "  Heard: $transcript"

    # Find and move the saved recording
    latest_wav=$(ls -t /tmp/voice_recording_*.wav 2>/dev/null | head -1)
    if [ -n "$latest_wav" ]; then
        mv "$latest_wav" "$OUTDIR/${padded}.wav"
        echo "  Saved: $OUTDIR/${padded}.wav"
    else
        echo "  WARNING: No recording saved"
    fi

done < "$PHRASES"

echo ""
echo "=== Recording session complete ==="
echo "Saved $n recordings to $OUTDIR/"
echo ""
echo "Run eval: ./eval/compare.sh"
