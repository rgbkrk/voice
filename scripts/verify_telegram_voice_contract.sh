#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

voice_bin="${VOICE_BIN:-}"
text="${TEXT:-Voice Telegram contract smoke test.}"
hermes_config="${HERMES_CONFIG:-${HOME:-}/.hermes/config.yaml}"
skip_hermes_config="${SKIP_HERMES_CONFIG:-0}"
skip_hermes_tts_smoke="${SKIP_HERMES_TTS_SMOKE:-0}"
require_daemon="${REQUIRE_DAEMON:-0}"
skip_daemon="${SKIP_DAEMON:-0}"
run_stt_smoke="${RUN_STT_SMOKE:-0}"
keep_output="${KEEP_OUTPUT:-0}"

voice_contract_script="${VOICE_CONTRACT_VERIFY_SCRIPT:-$repo_root/scripts/verify_whatsapp_voice_contract.sh}"
hermes_config_verify_script="${HERMES_CONFIG_VERIFY_SCRIPT:-$repo_root/scripts/verify_hermes_voice_config.py}"

usage() {
  cat <<'EOF'
Usage: scripts/verify_telegram_voice_contract.sh [OPTIONS]

Validate an installed voice binary against Telegram voice-message requirements.
This is an offline preflight; it does not require a Telegram bot token.

Options:
  --voice-bin PATH          voice binary to run (default: VOICE_BIN, target/release/voice, then PATH)
  --text TEXT               smoke text to synthesize
  --hermes-config PATH      Hermes config file to validate (default: HERMES_CONFIG or ~/.hermes/config.yaml)
  --skip-hermes-config      skip Hermes command-provider config validation
  --skip-hermes-tts-smoke   validate Hermes config without executing TTS
  --require-daemon          fail if voice daemon streaming is unavailable
  --skip-daemon             skip daemon-backed stream checks even if the daemon is running
  --run-stt-smoke           also replay a WAV through daemon stream-transcribe
  --keep-output             retain generated files from the underlying voice contract check
  -h, --help                show this help

Environment aliases:
  VOICE_BIN, TEXT, HERMES_CONFIG, SKIP_HERMES_CONFIG=1, SKIP_HERMES_TTS_SMOKE=1,
  REQUIRE_DAEMON=1, SKIP_DAEMON=1, RUN_STT_SMOKE=1, KEEP_OUTPUT=1
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
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
    --hermes-config)
      [[ $# -ge 2 ]] || fail "--hermes-config requires a path"
      hermes_config="$2"
      shift 2
      ;;
    --skip-hermes-config)
      skip_hermes_config=1
      shift
      ;;
    --skip-hermes-tts-smoke)
      skip_hermes_tts_smoke=1
      shift
      ;;
    --require-daemon)
      require_daemon=1
      shift
      ;;
    --skip-daemon)
      skip_daemon=1
      shift
      ;;
    --run-stt-smoke)
      run_stt_smoke=1
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

[[ -x "$voice_contract_script" ]] || fail "voice contract verifier is not executable: $voice_contract_script"

voice_contract_cmd=("$voice_contract_script" "--text" "$text")
if [[ -n "$voice_bin" ]]; then
  voice_contract_cmd+=("--voice-bin" "$voice_bin")
fi
if [[ "$require_daemon" == "1" ]]; then
  voice_contract_cmd+=("--require-daemon")
fi
if [[ "$skip_daemon" == "1" ]]; then
  voice_contract_cmd+=("--skip-daemon")
fi
if [[ "$run_stt_smoke" == "1" ]]; then
  voice_contract_cmd+=("--run-stt-smoke")
fi
if [[ "$keep_output" == "1" ]]; then
  voice_contract_cmd+=("--keep-output")
fi

"${voice_contract_cmd[@]}"
voice_contract_status="checked"

hermes_config_status="skipped"
if [[ "$skip_hermes_config" != "1" ]]; then
  [[ -x "$hermes_config_verify_script" ]] || fail "Hermes config verifier is not executable: $hermes_config_verify_script"
  [[ -f "$hermes_config" ]] || fail "Hermes config not found: $hermes_config"

  hermes_cmd=("$hermes_config_verify_script" "--config" "$hermes_config")
  if [[ -n "$voice_bin" ]]; then
    hermes_cmd+=("--voice-bin" "$voice_bin")
  fi
  if [[ "$skip_hermes_tts_smoke" == "1" ]]; then
    hermes_cmd+=("--skip-tts-smoke")
    hermes_config_status="checked_without_tts_smoke"
  else
    hermes_config_status="checked"
  fi

  "${hermes_cmd[@]}"
fi

echo "ok: voice Telegram contract verifier passed"
echo "voice_contract=$voice_contract_status"
echo "hermes_voice_config=$hermes_config_status"
echo "telegram_send_voice=compatible"
echo "telegram_credentials=not_required_for_offline_contract"
