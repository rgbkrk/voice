#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

voice_bin="${VOICE_BIN:-}"
python_bin="${PYTHON_BIN:-python3}"
host="${VOICE_WEBRTC_HOST:-127.0.0.1}"
port="${VOICE_WEBRTC_PORT:-8787}"
service_name="${VOICE_WEBRTC_SERVICE_NAME:-voice-webrtc-sidecar.service}"
log_level="${VOICE_WEBRTC_LOG_LEVEL:-INFO}"
venv="${VOICE_WEBRTC_VENV:-}"
rx_pcm="${VOICE_WEBRTC_RX_PCM:-}"
no_start=0
skip_venv=0
print_unit=0
uninstall=0

usage() {
  cat <<'EOF'
Usage: scripts/install_webrtc_sidecar_service.sh [OPTIONS]

Install the example WebRTC sidecar as a Linux systemd user service. The sidecar
is a localhost control plane for Hermes/WhatsApp Calling experiments and depends
on an installed voice daemon for the streaming contract.

Options:
  --repo-root PATH      voice repository root (default: script parent)
  --voice-bin PATH      voice binary exposed to the sidecar as VOICE_BIN
  --python-bin PATH     Python used to create the venv (default: python3)
  --venv PATH           sidecar venv path (default: XDG data dir)
  --host HOST           sidecar bind host; must be loopback (default: 127.0.0.1)
  --port PORT           sidecar bind port (default: 8787)
  --rx-pcm PATH         optional inbound PCM mirror path (default: XDG state dir)
  --service-name NAME   systemd user unit name (default: voice-webrtc-sidecar.service)
  --log-level LEVEL     sidecar log level (default: INFO)
  --no-start            install and enable the service without starting it
  --skip-venv           do not create/update the Python venv
  --print-unit          print the generated unit and exit without writing files
  --uninstall           disable and remove the systemd user unit
  -h, --help            show this help

Environment aliases:
  VOICE_BIN, PYTHON_BIN, VOICE_WEBRTC_VENV, VOICE_WEBRTC_HOST,
  VOICE_WEBRTC_PORT, VOICE_WEBRTC_RX_PCM, VOICE_WEBRTC_SERVICE_NAME,
  VOICE_WEBRTC_LOG_LEVEL
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required on PATH"
}

abs_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$(pwd)/$path"
  fi
}

unit_quote() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

default_data_home() {
  if [[ -n "${XDG_DATA_HOME:-}" ]]; then
    printf '%s\n' "$XDG_DATA_HOME"
  else
    printf '%s\n' "$HOME/.local/share"
  fi
}

default_state_home() {
  if [[ -n "${XDG_STATE_HOME:-}" ]]; then
    printf '%s\n' "$XDG_STATE_HOME"
  else
    printf '%s\n' "$HOME/.local/state"
  fi
}

default_config_home() {
  if [[ -n "${XDG_CONFIG_HOME:-}" ]]; then
    printf '%s\n' "$XDG_CONFIG_HOME"
  else
    printf '%s\n' "$HOME/.config"
  fi
}

resolve_voice_bin() {
  if [[ -n "$voice_bin" ]]; then
    if [[ "$voice_bin" == */* ]]; then
      abs_path "$voice_bin"
    else
      command -v "$voice_bin" || fail "voice binary not found on PATH: $voice_bin"
    fi
    return
  fi

  if command -v voice >/dev/null 2>&1; then
    command -v voice
  elif [[ -x "$repo_root/target/release/voice" ]]; then
    printf '%s\n' "$repo_root/target/release/voice"
  else
    fail "voice binary not found; pass --voice-bin PATH or install voice on PATH"
  fi
}

is_loopback_host() {
  case "$1" in
    localhost|127.*|::1)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_bind_host() {
  if is_loopback_host "$host"; then
    return
  fi
  fail "refusing non-loopback --host '$host'; the sidecar control plane must bind to localhost"
}

render_unit() {
  local sidecar_script="$repo_root/examples/webrtc-sidecar/sidecar.py"
  local python_exec="$venv/bin/python"

  cat <<EOF
[Unit]
Description=Voice WebRTC sidecar
After=voiced.service
Wants=voiced.service

[Service]
Type=simple
WorkingDirectory=$repo_root
Environment=$(unit_quote "VOICE_BIN=$voice_bin")
ExecStart=$(unit_quote "$python_exec") $(unit_quote "$sidecar_script") --host $(unit_quote "$host") --port $port --rx-pcm $(unit_quote "$rx_pcm") --log-level $(unit_quote "$log_level")
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      [[ $# -ge 2 ]] || fail "--repo-root requires a path"
      repo_root="$2"
      shift 2
      ;;
    --voice-bin)
      [[ $# -ge 2 ]] || fail "--voice-bin requires a path"
      voice_bin="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || fail "--python-bin requires a path"
      python_bin="$2"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || fail "--venv requires a path"
      venv="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || fail "--host requires a value"
      host="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || fail "--port requires a value"
      port="$2"
      shift 2
      ;;
    --rx-pcm)
      [[ $# -ge 2 ]] || fail "--rx-pcm requires a path"
      rx_pcm="$2"
      shift 2
      ;;
    --service-name)
      [[ $# -ge 2 ]] || fail "--service-name requires a value"
      service_name="$2"
      shift 2
      ;;
    --log-level)
      [[ $# -ge 2 ]] || fail "--log-level requires a value"
      log_level="$2"
      shift 2
      ;;
    --no-start)
      no_start=1
      shift
      ;;
    --skip-venv)
      skip_venv=1
      shift
      ;;
    --print-unit)
      print_unit=1
      shift
      ;;
    --uninstall)
      uninstall=1
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

[[ "$service_name" == *.service ]] || fail "--service-name must end in .service"
[[ "$service_name" != */* ]] || fail "--service-name must be a unit name, not a path"
[[ "$port" =~ ^[0-9]+$ ]] || fail "--port must be an integer"

unit_dir="$(default_config_home)/systemd/user"
unit_path="$unit_dir/$service_name"

if [[ "$uninstall" == "1" ]]; then
  require_command systemctl
  systemctl --user disable --now "$service_name" >/dev/null 2>&1 || true
  rm -f "$unit_path"
  systemctl --user daemon-reload >/dev/null 2>&1 || true
  echo "Uninstalled $service_name"
  echo "  removed: $unit_path"
  exit 0
fi

repo_root="$(cd "$repo_root" && pwd)"
sidecar_script="$repo_root/examples/webrtc-sidecar/sidecar.py"
requirements="$repo_root/examples/webrtc-sidecar/requirements.txt"
[[ -f "$sidecar_script" ]] || fail "sidecar script not found: $sidecar_script"
[[ -f "$requirements" ]] || fail "requirements file not found: $requirements"
validate_bind_host

if [[ -z "$venv" ]]; then
  venv="$(default_data_home)/voice/webrtc-sidecar-venv"
fi
if [[ -z "$rx_pcm" ]]; then
  rx_pcm="$(default_state_home)/voice/webrtc-sidecar/inbound.s16le"
fi

venv="$(abs_path "$venv")"
rx_pcm="$(abs_path "$rx_pcm")"

voice_bin="$(resolve_voice_bin)"

if [[ "$print_unit" == "1" ]]; then
  render_unit
  exit 0
fi

require_command systemctl
[[ -x "$voice_bin" ]] || fail "voice binary is not executable: $voice_bin"

if [[ "$skip_venv" != "1" ]]; then
  if [[ "$python_bin" == */* ]]; then
    [[ -x "$python_bin" ]] || fail "python binary is not executable: $python_bin"
  else
    python_bin="$(command -v "$python_bin" || true)"
    [[ -n "$python_bin" ]] || fail "python binary not found on PATH"
  fi

  mkdir -p "$(dirname "$venv")"
  "$python_bin" -m venv "$venv"
  "$venv/bin/pip" install -r "$requirements"
elif [[ ! -x "$venv/bin/python" ]]; then
  fail "--skip-venv was passed, but $venv/bin/python is not executable"
fi

mkdir -p "$unit_dir" "$(dirname "$rx_pcm")"
render_unit >"$unit_path"

systemctl --user daemon-reload
systemctl --user enable "$service_name" >/dev/null

if [[ "$no_start" != "1" ]]; then
  systemctl --user restart "$service_name"
fi

echo "Installed $service_name"
echo "  unit:    $unit_path"
echo "  voice:   $voice_bin"
echo "  venv:    $venv"
echo "  sidecar: http://$host:$port"
echo "  rx_pcm:  $rx_pcm"

if [[ "$no_start" != "1" ]]; then
  systemctl --user --no-pager --full status "$service_name" || true
fi
