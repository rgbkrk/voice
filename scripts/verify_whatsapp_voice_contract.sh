#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

voice_bin="${VOICE_BIN:-}"
text="${TEXT:-Voice WhatsApp contract smoke test.}"
keep_output="${KEEP_OUTPUT:-0}"
require_daemon="${REQUIRE_DAEMON:-0}"
skip_daemon="${SKIP_DAEMON:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/verify_whatsapp_voice_contract.sh [OPTIONS]

Validate an installed voice binary against the WhatsApp voice-note and
WebRTC/local-streaming contract.

Options:
  --voice-bin PATH     voice binary to run (default: VOICE_BIN, target/release/voice, then PATH)
  --text TEXT          smoke text to synthesize
  --require-daemon     fail if voice daemon streaming is unavailable
  --skip-daemon        skip daemon-backed stream checks even if the daemon is running
  --keep-output        retain generated files and print their directory
  -h, --help           show this help

Environment aliases:
  VOICE_BIN, TEXT, REQUIRE_DAEMON=1, SKIP_DAEMON=1, KEEP_OUTPUT=1
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required on PATH"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --voice-bin)
      [[ $# -ge 2 ]] || fail "--voice-bin requires a path"
      voice_bin="$2"
      shift 2
      ;;
    --text)
      [[ $# -ge 2 ]] || fail "--text requires a value"
      text="$2"
      shift 2
      ;;
    --require-daemon)
      require_daemon=1
      shift
      ;;
    --skip-daemon)
      skip_daemon=1
      shift
      ;;
    --keep-output)
      keep_output=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

if [[ "$require_daemon" == "1" && "$skip_daemon" == "1" ]]; then
  fail "--require-daemon and --skip-daemon cannot be combined"
fi

if [[ -z "$voice_bin" ]]; then
  if [[ -x "$repo_root/target/release/voice" ]]; then
    voice_bin="$repo_root/target/release/voice"
  else
    voice_bin="voice"
  fi
fi

require_command ffprobe
require_command python3

if [[ "$voice_bin" == */* ]]; then
  [[ -x "$voice_bin" ]] || fail "voice binary is not executable: $voice_bin"
else
  require_command "$voice_bin"
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/voice-whatsapp-contract.XXXXXX")"

cleanup() {
  if [[ "$keep_output" != "1" ]]; then
    rm -rf "$tmp_dir"
  fi
}
trap cleanup EXIT

probe_value() {
  local file="$1"
  local key="$2"
  ffprobe -v error \
    -select_streams a:0 \
    -show_entries stream=codec_name,sample_rate,channels \
    -of default=noprint_wrappers=1 \
    "$file" | sed -n "s/^${key}=//p" | head -n 1
}

assert_ogg_opus() {
  local file="$1"
  [[ -s "$file" ]] || fail "expected non-empty audio file: $file"
  local magic
  magic="$(head -c 4 "$file" || true)"
  [[ "$magic" == "OggS" ]] || fail "expected Ogg container magic for $file, got: ${magic:-<empty>}"

  local codec sample_rate channels
  codec="$(probe_value "$file" codec_name)"
  sample_rate="$(probe_value "$file" sample_rate)"
  channels="$(probe_value "$file" channels)"

  [[ "$codec" == "opus" ]] || fail "expected Opus codec for $file, got: ${codec:-<missing>}"
  [[ "$sample_rate" == "48000" ]] || fail "expected 48 kHz sample rate for $file, got: ${sample_rate:-<missing>}"
  [[ "$channels" == "1" ]] || fail "expected mono audio for $file, got channels=${channels:-<missing>}"
}

contract_path="$tmp_dir/stream-contract.json"
"$voice_bin" stream-contract >"$contract_path"

python3 - "$contract_path" "$repo_root/docs/contracts/webrtc-sidecar-v1.json" <<'PY'
import json
import pathlib
import sys

contract_path = pathlib.Path(sys.argv[1])
expected_path = pathlib.Path(sys.argv[2])
contract = json.loads(contract_path.read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(message)

require(contract.get("contract") == "voice.webrtc_sidecar", "bad contract id")
require(contract.get("version") == 1, "bad contract version")
audio = contract.get("audio") or {}
expected_audio = {
    "sample_rate": 48_000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le",
    "bytes_per_sample": 2,
    "samples_per_frame": 960,
    "frame_bytes": 1_920,
}
for key, expected in expected_audio.items():
    require(audio.get(key) == expected, f"audio.{key}={audio.get(key)!r}, expected {expected!r}")

surfaces = contract.get("voice_surfaces") or {}
for key in (
    "completed_voice_note",
    "streamed_voice_note",
    "raw_outbound_pcm",
    "raw_inbound_pcm",
    "file_transcription_smoke",
):
    require(key in surfaces, f"missing voice_surfaces.{key}")

require(
    surfaces["completed_voice_note"].get("output") == "audio/ogg; codecs=opus",
    "completed_voice_note output must be audio/ogg; codecs=opus",
)
require(
    surfaces["raw_outbound_pcm"].get("frame_bytes") == 1_920,
    "raw_outbound_pcm frame_bytes must be 1920",
)
require(
    surfaces["raw_inbound_pcm"].get("frame_bytes") == 1_920,
    "raw_inbound_pcm frame_bytes must be 1920",
)

endpoints = contract.get("endpoints") or {}
clear_audio = endpoints.get("clear_audio") or {}
require(
    clear_audio.get("method") == "POST",
    "clear_audio endpoint method must be POST",
)
require(
    clear_audio.get("path") == "/calls/{call_id}/audio/clear",
    "clear_audio endpoint path must be /calls/{call_id}/audio/clear",
)
payloads = contract.get("payloads") or {}
clear_response = payloads.get("clear_audio_response") or {}
require(
    "dropped_tx_bytes" in clear_response,
    "clear_audio_response must report dropped_tx_bytes",
)

if expected_path.is_file():
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    require(contract == expected, f"{contract_path} does not match {expected_path}")
PY

direct_ogg="$tmp_dir/reply.ogg"
"$voice_bin" --quiet say --format ogg-opus --output "$direct_ogg" "$text"
assert_ogg_opus "$direct_ogg"

inferred_ogg="$tmp_dir/inferred.ogg"
"$voice_bin" --quiet say --output "$inferred_ogg" "$text"
assert_ogg_opus "$inferred_ogg"

misleading_output="$tmp_dir/misleading-extension.txt"
if "$voice_bin" --quiet say --format ogg-opus --output "$tmp_dir/not-opus.wav" "$text" >"$misleading_output" 2>&1; then
  fail "voice accepted --format ogg-opus with a .wav output path"
fi
if [[ -e "$tmp_dir/not-opus.wav" ]]; then
  fail "voice created output despite misleading extension rejection"
fi

daemon_available=0
if "$voice_bin" daemon status >/dev/null 2>&1; then
  daemon_available=1
fi

if [[ "$skip_daemon" != "1" && "$daemon_available" == "1" ]]; then
  raw_pcm="$tmp_dir/stream.s16le"
  "$voice_bin" --quiet stream \
    --sample-rate 48000 \
    --frame-ms 20 \
    --raw-output "$raw_pcm" \
    "$text"
  python3 - "$raw_pcm" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
if not data:
    raise SystemExit("raw PCM stream is empty")
if len(data) % 1_920 != 0:
    raise SystemExit(f"raw PCM stream has {len(data)} bytes; expected whole 1920-byte frames")
if not any(data):
    raise SystemExit("raw PCM stream is silent")
PY

  streamed_ogg="$tmp_dir/streamed.ogg"
  "$voice_bin" --quiet stream \
    --sample-rate 48000 \
    --frame-ms 20 \
    --output "$streamed_ogg" \
    --format ogg-opus \
    "$text"
  assert_ogg_opus "$streamed_ogg"
  daemon_status="checked"
elif [[ "$require_daemon" == "1" ]]; then
  fail "voice daemon is not available; start it with 'voice daemon start --tts-only' or use --skip-daemon"
else
  daemon_status="skipped (daemon not running)"
fi

echo "ok: voice WhatsApp contract verifier passed"
echo "voice_bin=$voice_bin"
echo "daemon_streams=$daemon_status"
if [[ "$keep_output" == "1" ]]; then
  echo "contract=$contract_path"
  echo "direct_ogg=$direct_ogg"
  echo "inferred_ogg=$inferred_ogg"
  echo "output_dir=$tmp_dir"
else
  echo "output=<temporary; pass --keep-output to retain generated files>"
fi
