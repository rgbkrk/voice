#!/usr/bin/env bash
#
# Install `voice` with a stable code signature so the macOS microphone (TCC)
# grant survives rebuilds.
#
# Background: `cargo install --force` recompiles and ad-hoc-signs a fresh
# binary. macOS keys TCC grants on the code-signing identity (cdhash), so every
# ad-hoc rebuild looks like a new app and re-prompts for mic access. Signing
# with a stable identity (an Apple Development or Developer ID cert) keeps the
# identity constant across rebuilds, so the grant persists.
#
# Usage:
#   VOICE_SIGN_IDENTITY="Apple Development: you@example.com (TEAMID)" \
#     scripts/install_signed_voice.sh
#
#   # or pass the identity as the first argument
#   scripts/install_signed_voice.sh "Developer ID Application: Your Name (TEAMID)"
#
# Find your identities with:
#   security find-identity -v -p codesigning
#
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script is macOS-only (code signing is not needed elsewhere)" >&2
  exit 1
fi

IDENTITY="${1:-${VOICE_SIGN_IDENTITY:-}}"
if [[ -z "$IDENTITY" ]]; then
  echo "error: no signing identity given." >&2
  echo "  set VOICE_SIGN_IDENTITY or pass it as the first argument." >&2
  echo "  list identities: security find-identity -v -p codesigning" >&2
  exit 1
fi

BIN="${CARGO_HOME:-$HOME/.cargo}/bin/voice"

echo "Building and installing voice..."
cargo install --path "$(dirname "$0")/../crates/voice-cli" --force

if [[ ! -x "$BIN" ]]; then
  echo "error: expected installed binary at $BIN, not found" >&2
  exit 1
fi

ENTITLEMENTS="$(dirname "$0")/voice.entitlements"

echo "Signing $BIN with: $IDENTITY"
# --force replaces the ad-hoc signature cargo applied. --options runtime opts
# into the hardened runtime, which gates mic access behind the audio-input
# entitlement, so we pass voice.entitlements or the signed binary can't use the
# microphone.
codesign --force --options runtime \
  --entitlements "$ENTITLEMENTS" \
  --sign "$IDENTITY" "$BIN"

echo "Verifying signature..."
codesign --verify --verbose "$BIN"
codesign --display --verbose=2 "$BIN" 2>&1 | grep -E "Authority|TeamIdentifier|Identifier" || true
echo "Entitlements:"
codesign --display --entitlements - --xml "$BIN" 2>/dev/null | grep -q "audio-input" \
  && echo "  ✓ com.apple.security.device.audio-input present" \
  || echo "  ✗ audio-input entitlement missing — mic access will fail under hardened runtime"

echo
echo "Done. Now run:  voice daemon install"
echo "The mic permission prompt fires during install and, thanks to the stable"
echo "signature, won't re-prompt on future rebuilds signed with the same identity."
