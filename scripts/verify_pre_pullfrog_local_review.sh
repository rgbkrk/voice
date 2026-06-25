#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_ref="${BASE_REF:-origin/main}"
run_workspace="${RUN_WORKSPACE:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/verify_pre_pullfrog_local_review.sh [OPTIONS]

Run the local gate before moving a draft PR to ready for Pullfrog.

Options:
  --base REF       compare against REF for diff-smell checks (default: BASE_REF or origin/main)
  --workspace      run workspace tests and clippy instead of touched-package checks
  -h, --help       show this help

Environment:
  BASE_REF         default base ref for branch diff checks
  RUN_WORKSPACE=1 run workspace tests and clippy
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      [[ $# -ge 2 ]] || fail "--base requires a ref"
      base_ref="$2"
      shift 2
      ;;
    --workspace)
      run_workspace=1
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

cd "$repo_root"

git rev-parse --is-inside-work-tree >/dev/null || fail "not inside a git worktree"
merge_base="$(git merge-base "$base_ref" HEAD 2>/dev/null)" \
  || fail "could not find merge-base with $base_ref; fetch it or pass --base"

changed_files="$(
  {
    git diff --name-only "$merge_base"...HEAD
    git diff --name-only
    git diff --cached --name-only
  } | sort -u
)"

run cargo fmt --check
run git diff --check
run git diff --cached --check
run git diff --check "$merge_base"...HEAD

semantic_pathspec=(-- . ":(exclude)scripts/verify_pre_pullfrog_local_review.sh")
branch_diff="$(git diff -U0 "$merge_base"...HEAD "${semantic_pathspec[@]}")"
dirty_diff="$(git diff -U0 "${semantic_pathspec[@]}")"
staged_diff="$(git diff --cached -U0 "${semantic_pathspec[@]}")"
combined_diff="${branch_diff}
${dirty_diff}
${staged_diff}"

sentinel_zero_additions="$(
  printf '%s\n' "$combined_diff" \
    | grep -E '^\+[^+].*unwrap_or\(0\.0\)' \
    || true
)"
if [[ -n "$sentinel_zero_additions" ]]; then
  printf '%s\n' "$sentinel_zero_additions" >&2
  fail "new unwrap_or(0.0) measurement fallback found; use Option/default diagnostics or seed state explicitly"
fi

guard_additions="$(
  printf '%s\n' "$combined_diff" \
    | grep -E '^\+[^+].*(bail!|ensure!|return Err|Err\(|anyhow!\()' \
    | grep -Ev '^\+[^+].*(fn fail\(|fail \")' \
    || true
)"
test_additions="$(
  printf '%s\n' "$combined_diff" \
    | grep -E '^\+[^+].*(#\[test\]|fn [A-Za-z0-9_]*test|rejects_|errors_|fails_|assert_)' \
    || true
)"
if [[ -n "$guard_additions" && -z "$test_additions" ]]; then
  printf '%s\n' "$guard_additions" >&2
  fail "new error/guard path found without changed test assertions"
fi

packages=()
python_files=()
python_test_modules=()
run_python_helpers=0
add_package() {
  local package="$1"
  local existing
  for existing in "${packages[@]}"; do
    [[ "$existing" == "$package" ]] && return
  done
  packages+=("$package")
}

add_python_file() {
  local file="$1"
  local existing
  [[ -f "$file" ]] || return
  for existing in "${python_files[@]}"; do
    [[ "$existing" == "$file" ]] && return
  done
  python_files+=("$file")
}

add_python_test_module() {
  local module="$1"
  local existing
  for existing in "${python_test_modules[@]}"; do
    [[ "$existing" == "$module" ]] && return
  done
  python_test_modules+=("$module")
}

module_for_python_test() {
  local file="$1"
  file="${file%.py}"
  printf '%s\n' "${file//\//.}"
}

add_matching_python_test() {
  local file="$1"
  local base test_file
  case "$file" in
    tests/test_*.py)
      add_python_test_module "$(module_for_python_test "$file")"
      ;;
    scripts/*.py)
      base="$(basename "$file" .py)"
      test_file="tests/test_${base}.py"
      if [[ -f "$test_file" ]]; then
        add_python_file "$test_file"
        add_python_test_module "$(module_for_python_test "$test_file")"
      fi
      ;;
  esac
}

if [[ "$run_workspace" == "1" ]]; then
  packages=()
else
  while IFS= read -r file; do
    case "$file" in
      Cargo.toml|Cargo.lock)
        run_workspace=1
        ;;
      crates/voice-cli/*)
        add_package voice
        ;;
      crates/voice-eval/*)
        add_package voice-eval
        ;;
      crates/voice-voxtral/*)
        add_package voice-voxtral
        ;;
      crates/voice-tts/*)
        add_package voice-tts
        ;;
      crates/voice-stt/*)
        add_package voice-stt
        ;;
      crates/voice-whisper/*)
        add_package voice-whisper
        ;;
      crates/voice-g2p/*)
        add_package voice-g2p
        ;;
      crates/voice-audio/*)
        add_package voice-audio
        ;;
      crates/voice-stream/*)
        add_package voice-stream
        ;;
      crates/voice-protocol/*)
        add_package voice-protocol
        ;;
      crates/voice-daemon/*)
        add_package voice-daemon
        ;;
      scripts/*.py|tests/*.py)
        add_python_file "$file"
        add_matching_python_test "$file"
        run_python_helpers=1
        ;;
      .github/workflows/*.yml|.github/workflows/*.yaml)
        run_python_helpers=1
        ;;
    esac
  done <<< "$changed_files"
fi

if [[ "$run_workspace" == "1" ]]; then
  run cargo test --workspace
  run cargo clippy --workspace --all-targets -- -D warnings
elif [[ "${#packages[@]}" -gt 0 ]]; then
  for package in "${packages[@]}"; do
    run cargo test -p "$package"
  done
  clippy_args=()
  for package in "${packages[@]}"; do
    clippy_args+=("-p" "$package")
  done
  run cargo clippy "${clippy_args[@]}" --all-targets -- -D warnings
else
  echo "No Rust package changes detected; skipped cargo test/clippy package gates."
fi

if [[ "$run_python_helpers" == "1" ]]; then
  if [[ "${#python_files[@]}" -gt 0 ]]; then
    run python3 -m py_compile "${python_files[@]}"
  fi
  if [[ "${#python_test_modules[@]}" -gt 0 ]]; then
    run python3 -m unittest "${python_test_modules[@]}"
  else
    echo "No matching Python test modules detected; skipped Python unittest gate."
  fi
fi

cat <<EOF

Pre-Pullfrog local gate passed.

Still do the human/agent review pass before undrafting:
- compare PR body claims against the actual diff and artifact paths
- enumerate every changed user-facing surface
- check for duplicated helper logic near new helpers
- record the local evidence in the PR body
EOF
