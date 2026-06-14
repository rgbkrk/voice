#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

voice_bin="${VOICE_BIN:-}"
hermes_config="${HERMES_CONFIG:-${HOME:-}/.hermes/config.yaml}"
hermes_home="${HERMES_HOME:-${HOME:-}/.hermes}"
sidecar_url="${SIDECAR_URL:-http://127.0.0.1:8787}"
text="${TEXT:-Local Hermes voice stack smoke test.}"
skip_hermes_config="${SKIP_HERMES_CONFIG:-0}"
skip_hermes_tts_smoke="${SKIP_HERMES_TTS_SMOKE:-0}"
skip_hermes_gateway="${SKIP_HERMES_GATEWAY:-0}"
skip_sidecar="${SKIP_SIDECAR:-0}"
skip_whatsapp_bridge="${SKIP_WHATSAPP_BRIDGE:-0}"
skip_systemd="${SKIP_SYSTEMD:-0}"
skip_cli_mcp="${SKIP_CLI_MCP:-0}"
skip_daemon="${SKIP_DAEMON:-0}"
skip_stt_smoke="${SKIP_STT_SMOKE:-0}"
run_whatsapp_inbound_cache_smoke="${RUN_WHATSAPP_INBOUND_CACHE_SMOKE:-0}"
whatsapp_alpha_profile="${WHATSAPP_ALPHA_PROFILE:-}"
whatsapp_alpha_voice_note_chat_id="${WHATSAPP_ALPHA_VOICE_NOTE_CHAT_ID:-}"
whatsapp_alpha_wait_audio_cache_seconds="${WHATSAPP_ALPHA_WAIT_AUDIO_CACHE_SECONDS:-}"
whatsapp_alpha_wait_inbound_seconds="${WHATSAPP_ALPHA_WAIT_INBOUND_SECONDS:-}"
run_webrtc_loopback_smoke="${RUN_WEBRTC_LOOPBACK_SMOKE:-0}"
webrtc_python="${VOICE_WEBRTC_PYTHON:-python3}"
webrtc_timeout="${VOICE_WEBRTC_TIMEOUT:-60}"
max_queued_tx_ms="${MAX_QUEUED_TX_MS:-1000}"
whatsapp_bridge_url="${WHATSAPP_BRIDGE_URL:-http://127.0.0.1:3000}"
whatsapp_session_dir="${WHATSAPP_SESSION_DIR:-}"
whatsapp_env_file="${WHATSAPP_ENV_FILE:-}"
whatsapp_audio_cache_dir="${WHATSAPP_AUDIO_CACHE_DIR:-}"
expected_whatsapp_agent_number="${WHATSAPP_AGENT_NUMBER:-}"
expected_whatsapp_agent_name="${WHATSAPP_AGENT_NAME:-}"
require_whatsapp_cloud="${REQUIRE_WHATSAPP_CLOUD:-0}"
require_whatsapp_calling="${REQUIRE_WHATSAPP_CALLING:-0}"

hermes_config_verify_script="${HERMES_CONFIG_VERIFY_SCRIPT:-$repo_root/scripts/verify_hermes_voice_config.py}"
hermes_gateway_verify_script="${HERMES_GATEWAY_VERIFY_SCRIPT:-$repo_root/scripts/verify_hermes_gateway_service.py}"
cli_mcp_surface_verify_script="${CLI_MCP_SURFACE_VERIFY_SCRIPT:-$repo_root/scripts/verify_cli_mcp_surface.py}"
whatsapp_contract_verify_script="${WHATSAPP_CONTRACT_VERIFY_SCRIPT:-$repo_root/scripts/verify_whatsapp_voice_contract.sh}"
whatsapp_bridge_verify_script="${WHATSAPP_BRIDGE_VERIFY_SCRIPT:-$repo_root/scripts/verify_whatsapp_bridge_runtime.py}"
whatsapp_inbound_cache_verify_script="${WHATSAPP_INBOUND_CACHE_VERIFY_SCRIPT:-$repo_root/scripts/verify_whatsapp_inbound_audio_cache.py}"
whatsapp_alpha_readiness_script="${WHATSAPP_ALPHA_READINESS_SCRIPT:-$repo_root/scripts/verify_whatsapp_alpha_readiness.py}"
sidecar_service_verify_script="${SIDECAR_SERVICE_VERIFY_SCRIPT:-$repo_root/scripts/verify_webrtc_sidecar_service.py}"
webrtc_loopback_smoke_script="${WEBRTC_LOOPBACK_SMOKE_SCRIPT:-$repo_root/examples/webrtc-sidecar/full_duplex_loopback_smoke.py}"

usage() {
  cat <<'EOF'
Usage: scripts/verify_local_hermes_voice_stack.sh [OPTIONS]

Run the local release gate for a Hermes host that uses voice for WhatsApp-ready
TTS/STT and WebRTC sidecar streaming.

By default this requires the voice daemon, runs the stream-transcribe smoke, and
checks the WebRTC sidecar HTTP contract plus Linux systemd user services.

Options:
  --voice-bin PATH             installed voice binary to verify
  --hermes-config PATH         Hermes config file (default: ~/.hermes/config.yaml)
  --hermes-home PATH           Hermes home directory (default: ~/.hermes)
  --sidecar-url URL            sidecar base URL (default: http://127.0.0.1:8787)
  --text TEXT                  smoke text used by TTS checks
  --skip-hermes-config         skip Hermes config validation and TTS command smoke
  --skip-hermes-tts-smoke      validate Hermes config without executing TTS
  --skip-hermes-gateway        skip running Hermes gateway service verification
  --skip-cli-mcp               skip plain CLI/MCP daemon surface verification
  --skip-whatsapp-bridge       skip local WhatsApp bridge identity verification
  --skip-sidecar               skip WebRTC sidecar service verification
  --skip-systemd               skip Hermes gateway, sidecar, and daemon systemd service checks
  --skip-daemon                skip daemon-backed stream checks
  --skip-stt-smoke             skip stream-transcribe smoke
  --whatsapp-bridge-url URL    Baileys bridge URL (default: http://127.0.0.1:3000)
  --whatsapp-session-dir PATH  Baileys multi-file auth session directory
  --whatsapp-env-file PATH     Hermes env file with WhatsApp settings
  --whatsapp-audio-cache-dir PATH
                               Hermes/bridge audio cache directory
  --expected-whatsapp-agent-number NUMBER
                               require Baileys creds to be paired to this number
  --expected-whatsapp-agent-name NAME
                               require Baileys creds to expose this profile name
  --require-whatsapp-cloud     fail when WhatsApp Cloud credentials are missing
  --require-whatsapp-calling   fail when Cloud Calling credentials/readiness are missing
  --run-whatsapp-inbound-cache-smoke
                               transcribe a bridge-downloaded aud_* file from the audio cache
  --whatsapp-alpha-profile PROFILE
                               run categorized alpha readiness profile:
                               unattended, cached-receive, send,
                               attended-cache-receive, attended-send-receive
  --whatsapp-alpha-chat-id ID  override WHATSAPP_HOME_CHANNEL for alpha sends
  --whatsapp-alpha-wait-audio-cache-seconds SECONDS
                               override attended-cache-receive cache watch time
  --whatsapp-alpha-wait-inbound-seconds SECONDS
                               override attended-send-receive bridge poll time
  --run-webrtc-loopback-smoke  run one local full-duplex WebRTC media turn
  --webrtc-python PATH         Python used for the WebRTC smoke (default: python3)
  --webrtc-timeout SECONDS     timeout passed to the WebRTC smoke (default: 60)
  --max-queued-tx-ms MS        max sidecar queue after WebRTC smoke (default: 1000)
  -h, --help                   show this help

Environment aliases:
  VOICE_BIN, HERMES_CONFIG, SIDECAR_URL, TEXT
  SKIP_HERMES_CONFIG=1, SKIP_HERMES_TTS_SMOKE=1, SKIP_SIDECAR=1
  SKIP_HERMES_GATEWAY=1, SKIP_WHATSAPP_BRIDGE=1, SKIP_SYSTEMD=1, SKIP_CLI_MCP=1
  SKIP_DAEMON=1, SKIP_STT_SMOKE=1, RUN_WEBRTC_LOOPBACK_SMOKE=1
  WHATSAPP_BRIDGE_URL, WHATSAPP_SESSION_DIR, WHATSAPP_ENV_FILE
  WHATSAPP_AUDIO_CACHE_DIR, WHATSAPP_AGENT_NUMBER, WHATSAPP_AGENT_NAME
  REQUIRE_WHATSAPP_CLOUD=1, REQUIRE_WHATSAPP_CALLING=1
  RUN_WHATSAPP_INBOUND_CACHE_SMOKE=1, WHATSAPP_ALPHA_PROFILE
  WHATSAPP_ALPHA_VOICE_NOTE_CHAT_ID, WHATSAPP_ALPHA_WAIT_AUDIO_CACHE_SECONDS
  WHATSAPP_ALPHA_WAIT_INBOUND_SECONDS
  VOICE_WEBRTC_PYTHON, VOICE_WEBRTC_TIMEOUT
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || fail "$label not found: $path"
}

require_executable() {
  local path="$1"
  local label="$2"
  [[ -x "$path" ]] || fail "$label is not executable: $path"
}

default_voice_bin() {
  if [[ -n "$voice_bin" ]]; then
    printf '%s\n' "$voice_bin"
  elif command -v voice >/dev/null 2>&1; then
    command -v voice
  elif [[ -x "$repo_root/target/release/voice" ]]; then
    printf '%s\n' "$repo_root/target/release/voice"
  else
    printf '%s\n' "voice"
  fi
}

run_step() {
  local label="$1"
  shift
  echo
  echo "==> $label"
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --voice-bin)
      [[ $# -ge 2 ]] || fail "--voice-bin requires a path"
      voice_bin="$2"
      shift 2
      ;;
    --hermes-config)
      [[ $# -ge 2 ]] || fail "--hermes-config requires a path"
      hermes_config="$2"
      shift 2
      ;;
    --hermes-home)
      [[ $# -ge 2 ]] || fail "--hermes-home requires a path"
      hermes_home="$2"
      shift 2
      ;;
    --sidecar-url)
      [[ $# -ge 2 ]] || fail "--sidecar-url requires a URL"
      sidecar_url="$2"
      shift 2
      ;;
    --text)
      [[ $# -ge 2 ]] || fail "--text requires a value"
      text="$2"
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
    --skip-hermes-gateway)
      skip_hermes_gateway=1
      shift
      ;;
    --skip-sidecar)
      skip_sidecar=1
      shift
      ;;
    --skip-whatsapp-bridge)
      skip_whatsapp_bridge=1
      shift
      ;;
    --skip-cli-mcp)
      skip_cli_mcp=1
      shift
      ;;
    --skip-systemd)
      skip_systemd=1
      shift
      ;;
    --skip-daemon)
      skip_daemon=1
      shift
      ;;
    --skip-stt-smoke)
      skip_stt_smoke=1
      shift
      ;;
    --whatsapp-bridge-url)
      [[ $# -ge 2 ]] || fail "--whatsapp-bridge-url requires a URL"
      whatsapp_bridge_url="$2"
      shift 2
      ;;
    --whatsapp-session-dir)
      [[ $# -ge 2 ]] || fail "--whatsapp-session-dir requires a path"
      whatsapp_session_dir="$2"
      shift 2
      ;;
    --whatsapp-env-file)
      [[ $# -ge 2 ]] || fail "--whatsapp-env-file requires a path"
      whatsapp_env_file="$2"
      shift 2
      ;;
    --whatsapp-audio-cache-dir)
      [[ $# -ge 2 ]] || fail "--whatsapp-audio-cache-dir requires a path"
      whatsapp_audio_cache_dir="$2"
      shift 2
      ;;
    --expected-whatsapp-agent-number)
      [[ $# -ge 2 ]] || fail "--expected-whatsapp-agent-number requires a value"
      expected_whatsapp_agent_number="$2"
      shift 2
      ;;
    --expected-whatsapp-agent-name)
      [[ $# -ge 2 ]] || fail "--expected-whatsapp-agent-name requires a value"
      expected_whatsapp_agent_name="$2"
      shift 2
      ;;
    --require-whatsapp-cloud)
      require_whatsapp_cloud=1
      shift
      ;;
    --require-whatsapp-calling)
      require_whatsapp_calling=1
      shift
      ;;
    --run-whatsapp-inbound-cache-smoke)
      run_whatsapp_inbound_cache_smoke=1
      shift
      ;;
    --whatsapp-alpha-profile)
      [[ $# -ge 2 ]] || fail "--whatsapp-alpha-profile requires a profile"
      whatsapp_alpha_profile="$2"
      shift 2
      ;;
    --whatsapp-alpha-chat-id)
      [[ $# -ge 2 ]] || fail "--whatsapp-alpha-chat-id requires a chat id"
      whatsapp_alpha_voice_note_chat_id="$2"
      shift 2
      ;;
    --whatsapp-alpha-wait-audio-cache-seconds)
      [[ $# -ge 2 ]] || fail "--whatsapp-alpha-wait-audio-cache-seconds requires a value"
      whatsapp_alpha_wait_audio_cache_seconds="$2"
      shift 2
      ;;
    --whatsapp-alpha-wait-inbound-seconds)
      [[ $# -ge 2 ]] || fail "--whatsapp-alpha-wait-inbound-seconds requires a value"
      whatsapp_alpha_wait_inbound_seconds="$2"
      shift 2
      ;;
    --run-webrtc-loopback-smoke)
      run_webrtc_loopback_smoke=1
      shift
      ;;
    --webrtc-python)
      [[ $# -ge 2 ]] || fail "--webrtc-python requires a path"
      webrtc_python="$2"
      shift 2
      ;;
    --webrtc-timeout)
      [[ $# -ge 2 ]] || fail "--webrtc-timeout requires a value"
      webrtc_timeout="$2"
      shift 2
      ;;
    --max-queued-tx-ms)
      [[ $# -ge 2 ]] || fail "--max-queued-tx-ms requires a value"
      max_queued_tx_ms="$2"
      shift 2
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

if [[ -n "$whatsapp_alpha_profile" ]]; then
  case "$whatsapp_alpha_profile" in
    unattended|cached-receive|send|attended-cache-receive|attended-send-receive)
      ;;
    *)
      fail "unknown WhatsApp alpha profile: $whatsapp_alpha_profile"
      ;;
  esac
fi

voice_bin="$(default_voice_bin)"
if [[ "$voice_bin" == */* ]]; then
  require_executable "$voice_bin" "voice binary"
elif ! command -v "$voice_bin" >/dev/null 2>&1; then
  fail "voice binary not found on PATH: $voice_bin"
fi

require_executable "$whatsapp_contract_verify_script" "WhatsApp voice contract verifier"
if [[ "$skip_whatsapp_bridge" != "1" ]]; then
  require_executable "$whatsapp_bridge_verify_script" "WhatsApp bridge runtime verifier"
fi
if [[ "$run_whatsapp_inbound_cache_smoke" == "1" ]]; then
  require_executable "$whatsapp_inbound_cache_verify_script" "WhatsApp inbound audio cache verifier"
fi
if [[ -n "$whatsapp_alpha_profile" ]]; then
  require_executable "$whatsapp_alpha_readiness_script" "WhatsApp alpha readiness verifier"
fi
if [[ "$skip_hermes_config" != "1" ]]; then
  require_executable "$hermes_config_verify_script" "Hermes voice config verifier"
  require_file "$hermes_config" "Hermes config"
fi
if [[ "$skip_cli_mcp" != "1" ]]; then
  require_executable "$cli_mcp_surface_verify_script" "CLI/MCP surface verifier"
fi
if [[ "$skip_hermes_gateway" != "1" && "$skip_systemd" != "1" ]]; then
  require_executable "$hermes_gateway_verify_script" "Hermes gateway service verifier"
fi
if [[ "$skip_sidecar" != "1" ]]; then
  require_executable "$sidecar_service_verify_script" "WebRTC sidecar verifier"
fi
if [[ "$run_webrtc_loopback_smoke" == "1" ]]; then
  require_file "$webrtc_loopback_smoke_script" "WebRTC full-duplex smoke"
  if [[ "$webrtc_python" == */* ]]; then
    require_executable "$webrtc_python" "WebRTC smoke Python"
  elif ! command -v "$webrtc_python" >/dev/null 2>&1; then
    fail "WebRTC smoke Python not found on PATH: $webrtc_python"
  fi
fi

if [[ "$skip_hermes_config" != "1" ]]; then
  hermes_args=(
    "$hermes_config_verify_script"
    --config "$hermes_config"
    --voice-bin "$voice_bin"
    --text "$text"
  )
  if [[ "$skip_hermes_tts_smoke" == "1" ]]; then
    hermes_args+=(--skip-tts-smoke)
  fi
  run_step "Hermes voice-native config" "${hermes_args[@]}"
  hermes_status="checked"
else
  hermes_status="skipped"
fi

if [[ "$skip_hermes_gateway" != "1" && "$skip_systemd" != "1" ]]; then
  gateway_args=(
    "$hermes_gateway_verify_script"
    --voice-bin "$voice_bin"
    --hermes-home "$hermes_home"
    --sidecar-url "$sidecar_url"
  )
  run_step "Hermes gateway voice stream service" "${gateway_args[@]}"
  hermes_gateway_status="checked"
else
  hermes_gateway_status="skipped"
fi

if [[ "$skip_cli_mcp" != "1" ]]; then
  cli_mcp_args=(
    "$cli_mcp_surface_verify_script"
    --voice-bin "$voice_bin"
  )
  if [[ "$skip_daemon" == "1" ]]; then
    cli_mcp_args+=(--skip-daemon)
  else
    cli_mcp_args+=(--require-daemon)
  fi
  run_step "Voice CLI and MCP daemon surfaces" "${cli_mcp_args[@]}"
  cli_mcp_status="checked"
else
  cli_mcp_status="skipped"
fi

whatsapp_args=(
  "$whatsapp_contract_verify_script"
  --voice-bin "$voice_bin"
  --text "$text"
)
if [[ "$skip_daemon" == "1" ]]; then
  whatsapp_args+=(--skip-daemon)
else
  whatsapp_args+=(--require-daemon)
  if [[ "$skip_stt_smoke" != "1" ]]; then
    whatsapp_args+=(--run-stt-smoke)
  fi
fi
run_step "Voice WhatsApp and streaming contract" "${whatsapp_args[@]}"

if [[ "$skip_whatsapp_bridge" != "1" ]]; then
  whatsapp_bridge_args=(
    "$whatsapp_bridge_verify_script"
    --hermes-home "$hermes_home"
    --bridge-url "$whatsapp_bridge_url"
  )
  if [[ -n "$whatsapp_session_dir" ]]; then
    whatsapp_bridge_args+=(--session-dir "$whatsapp_session_dir")
  fi
  if [[ -n "$whatsapp_env_file" ]]; then
    whatsapp_bridge_args+=(--env-file "$whatsapp_env_file")
  fi
  if [[ -n "$expected_whatsapp_agent_number" ]]; then
    whatsapp_bridge_args+=(--expected-agent-number "$expected_whatsapp_agent_number")
  fi
  if [[ -n "$expected_whatsapp_agent_name" ]]; then
    whatsapp_bridge_args+=(--expected-agent-name "$expected_whatsapp_agent_name")
  fi
  if [[ "$skip_systemd" == "1" ]]; then
    whatsapp_bridge_args+=(--skip-systemd)
  fi
  if [[ "$require_whatsapp_cloud" == "1" ]]; then
    whatsapp_bridge_args+=(--require-whatsapp-cloud)
  fi
  if [[ "$require_whatsapp_calling" == "1" ]]; then
    whatsapp_bridge_args+=(--require-whatsapp-calling)
  fi
  run_step "WhatsApp bridge identity and credential readiness" "${whatsapp_bridge_args[@]}"
  whatsapp_bridge_status="checked"
else
  whatsapp_bridge_status="skipped"
fi

if [[ "$run_whatsapp_inbound_cache_smoke" == "1" ]]; then
  whatsapp_inbound_cache_args=(
    "$whatsapp_inbound_cache_verify_script"
    --voice-bin "$voice_bin"
    --hermes-home "$hermes_home"
    --require-cache
    --run-stt
  )
  if [[ -n "$whatsapp_audio_cache_dir" ]]; then
    whatsapp_inbound_cache_args+=(--audio-cache-dir "$whatsapp_audio_cache_dir")
  fi
  run_step "WhatsApp inbound cached audio STT smoke" "${whatsapp_inbound_cache_args[@]}"
  whatsapp_inbound_cache_status="checked"
else
  whatsapp_inbound_cache_status="skipped"
fi

if [[ "$skip_sidecar" != "1" ]]; then
  sidecar_args=(
    "$sidecar_service_verify_script"
    --voice-bin "$voice_bin"
    --sidecar-url "$sidecar_url"
  )
  if [[ "$skip_systemd" == "1" ]]; then
    sidecar_args+=(--skip-systemd)
  fi
  run_step "Voice WebRTC sidecar service" "${sidecar_args[@]}"
  sidecar_status="checked"
else
  sidecar_status="skipped"
fi

if [[ "$run_webrtc_loopback_smoke" == "1" ]]; then
  run_step "Full-duplex WebRTC media smoke" \
    "$webrtc_python" "$webrtc_loopback_smoke_script" \
    --voice-bin "$voice_bin" \
    --sidecar-url "$sidecar_url" \
    --timeout "$webrtc_timeout" \
    --outbound-text "$text" \
    --max-queued-tx-ms "$max_queued_tx_ms"
  webrtc_loopback_status="checked"
else
  webrtc_loopback_status="skipped"
fi

if [[ -n "$whatsapp_alpha_profile" ]]; then
  whatsapp_alpha_args=(
    "$whatsapp_alpha_readiness_script"
    --voice-bin "$voice_bin"
    --hermes-home "$hermes_home"
    --hermes-config "$hermes_config"
    --bridge-url "$whatsapp_bridge_url"
    --sidecar-url "$sidecar_url"
    --profile "$whatsapp_alpha_profile"
    --text "$text"
  )
  if [[ -n "$whatsapp_audio_cache_dir" ]]; then
    whatsapp_alpha_args+=(--whatsapp-audio-cache-dir "$whatsapp_audio_cache_dir")
  fi
  if [[ -n "$whatsapp_alpha_voice_note_chat_id" ]]; then
    whatsapp_alpha_args+=(--voice-note-chat-id "$whatsapp_alpha_voice_note_chat_id")
  fi
  if [[ -n "$whatsapp_alpha_wait_audio_cache_seconds" ]]; then
    whatsapp_alpha_args+=(--wait-audio-cache-seconds "$whatsapp_alpha_wait_audio_cache_seconds")
  fi
  if [[ -n "$whatsapp_alpha_wait_inbound_seconds" ]]; then
    whatsapp_alpha_args+=(--wait-inbound-seconds "$whatsapp_alpha_wait_inbound_seconds")
  fi
  if [[ -n "$expected_whatsapp_agent_number" ]]; then
    whatsapp_alpha_args+=(--expected-agent-number "$expected_whatsapp_agent_number")
  fi
  if [[ -n "$expected_whatsapp_agent_name" ]]; then
    whatsapp_alpha_args+=(--expected-agent-name "$expected_whatsapp_agent_name")
  fi
  if [[ "$skip_systemd" == "1" ]]; then
    whatsapp_alpha_args+=(--skip-systemd)
  fi
  if [[ "$skip_daemon" == "1" ]]; then
    whatsapp_alpha_args+=(--skip-daemon)
  elif [[ "$skip_stt_smoke" != "1" ]]; then
    whatsapp_alpha_args+=(--run-stt-smoke)
  fi
  if [[ "$skip_sidecar" == "1" ]]; then
    whatsapp_alpha_args+=(--skip-sidecar)
  fi
  if [[ "$skip_hermes_tts_smoke" == "1" ]]; then
    whatsapp_alpha_args+=(--skip-hermes-tts-smoke)
  fi
  if [[ "$require_whatsapp_cloud" == "1" ]]; then
    whatsapp_alpha_args+=(--require-whatsapp-cloud)
  fi
  if [[ "$require_whatsapp_calling" == "1" ]]; then
    whatsapp_alpha_args+=(--require-whatsapp-calling)
  fi
  run_step "WhatsApp alpha readiness profile ($whatsapp_alpha_profile)" "${whatsapp_alpha_args[@]}"
  whatsapp_alpha_status="$whatsapp_alpha_profile"
else
  whatsapp_alpha_status="skipped"
fi

echo
echo "ok: local Hermes voice stack verifier passed"
echo "voice_bin=$voice_bin"
echo "hermes_config=$hermes_status"
echo "hermes_gateway=$hermes_gateway_status"
echo "cli_mcp=$cli_mcp_status"
echo "whatsapp_contract=checked"
echo "whatsapp_bridge=$whatsapp_bridge_status"
echo "whatsapp_inbound_cache=$whatsapp_inbound_cache_status"
echo "whatsapp_alpha=$whatsapp_alpha_status"
echo "sidecar_service=$sidecar_status"
echo "webrtc_loopback=$webrtc_loopback_status"
