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
skip_systemd="${SKIP_SYSTEMD:-0}"
skip_cli_mcp="${SKIP_CLI_MCP:-0}"
skip_daemon="${SKIP_DAEMON:-0}"
skip_stt_smoke="${SKIP_STT_SMOKE:-0}"
run_webrtc_loopback_smoke="${RUN_WEBRTC_LOOPBACK_SMOKE:-0}"
webrtc_python="${VOICE_WEBRTC_PYTHON:-python3}"
webrtc_timeout="${VOICE_WEBRTC_TIMEOUT:-60}"
max_queued_tx_ms="${MAX_QUEUED_TX_MS:-1000}"

hermes_config_verify_script="${HERMES_CONFIG_VERIFY_SCRIPT:-$repo_root/scripts/verify_hermes_voice_config.py}"
hermes_gateway_verify_script="${HERMES_GATEWAY_VERIFY_SCRIPT:-$repo_root/scripts/verify_hermes_gateway_service.py}"
cli_mcp_surface_verify_script="${CLI_MCP_SURFACE_VERIFY_SCRIPT:-$repo_root/scripts/verify_cli_mcp_surface.py}"
whatsapp_contract_verify_script="${WHATSAPP_CONTRACT_VERIFY_SCRIPT:-$repo_root/scripts/verify_whatsapp_voice_contract.sh}"
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
  --skip-sidecar               skip WebRTC sidecar service verification
  --skip-systemd               skip Hermes gateway, sidecar, and daemon systemd service checks
  --skip-daemon                skip daemon-backed stream checks
  --skip-stt-smoke             skip stream-transcribe smoke
  --run-webrtc-loopback-smoke  run one local full-duplex WebRTC media turn
  --webrtc-python PATH         Python used for the WebRTC smoke (default: python3)
  --webrtc-timeout SECONDS     timeout passed to the WebRTC smoke (default: 60)
  --max-queued-tx-ms MS        max sidecar queue after WebRTC smoke (default: 1000)
  -h, --help                   show this help

Environment aliases:
  VOICE_BIN, HERMES_CONFIG, SIDECAR_URL, TEXT
  SKIP_HERMES_CONFIG=1, SKIP_HERMES_TTS_SMOKE=1, SKIP_SIDECAR=1
  SKIP_HERMES_GATEWAY=1, SKIP_SYSTEMD=1, SKIP_CLI_MCP=1
  SKIP_DAEMON=1, SKIP_STT_SMOKE=1, RUN_WEBRTC_LOOPBACK_SMOKE=1
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

voice_bin="$(default_voice_bin)"
if [[ "$voice_bin" == */* ]]; then
  require_executable "$voice_bin" "voice binary"
elif ! command -v "$voice_bin" >/dev/null 2>&1; then
  fail "voice binary not found on PATH: $voice_bin"
fi

require_executable "$whatsapp_contract_verify_script" "WhatsApp voice contract verifier"
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
    --timeout "$webrtc_timeout" \
    --outbound-text "$text" \
    --max-queued-tx-ms "$max_queued_tx_ms"
  webrtc_loopback_status="checked"
else
  webrtc_loopback_status="skipped"
fi

echo
echo "ok: local Hermes voice stack verifier passed"
echo "voice_bin=$voice_bin"
echo "hermes_config=$hermes_status"
echo "hermes_gateway=$hermes_gateway_status"
echo "cli_mcp=$cli_mcp_status"
echo "whatsapp_contract=checked"
echo "sidecar_service=$sidecar_status"
echo "webrtc_loopback=$webrtc_loopback_status"
