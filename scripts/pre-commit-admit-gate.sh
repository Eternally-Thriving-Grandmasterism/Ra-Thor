#!/usr/bin/env bash
# pre-commit-admit-gate.sh — Local Tier A admission gate for Ra-Thor / mercy-security
#
# Runs `mercy-admit` (or builds it via cargo) against staged text-like files.
# Exit non-zero on Medium+ so the commit is blocked for unattended paths.
#
# Usage (git hook):
#   ln -sf ../../scripts/pre-commit-admit-gate.sh .git/hooks/pre-commit
#   # or call from a pre-commit framework:
#   ./scripts/pre-commit-admit-gate.sh
#
# Escape hatch (human-reviewed only, never default on main):
#   WHITEHAT_ALLOW_MEDIUM=1 ./scripts/pre-commit-admit-gate.sh
#
# Contact: info@Rathor.ai | AG-SML v1.0 | White-hat only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional: only scan these extensions / names (expand as needed)
EXTENSIONS_REGEX='\.(md|txt|py|yml|yaml|json|toml|rs|sh|js|ts|jsx|tsx|ipynb)$'

collect_staged() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --cached --name-only --diff-filter=ACMR | grep -E "$EXTENSIONS_REGEX" || true
  else
    # Fallback when not in a git repo: scan fixtures for smoke
    find crates/mercy-security/fixtures -type f 2>/dev/null || true
  fi
}

FILES=()
while IFS= read -r f; do
  [[ -n "$f" && -f "$f" ]] && FILES+=("$f")
done < <(collect_staged)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "[mercy-admit] no staged text-like files — skip"
  exit 0
fi

echo "[mercy-admit] scanning ${#FILES[@]} staged file(s)..."

# Prefer already-built binary; otherwise cargo run (uses workspace cache)
BIN="target/debug/mercy-admit"
if [[ ! -x "$BIN" ]]; then
  BIN="target/release/mercy-admit"
fi

if [[ -x "$BIN" ]]; then
  CMD=("$BIN")
else
  echo "[mercy-admit] building via cargo (first run may take a moment)..."
  CMD=(cargo run -q -p mercy-security --bin mercy-admit --)
fi

# Always pass --verbose so findings appear in the hook output
if "${CMD[@]}" --verbose "${FILES[@]}"; then
  echo "[mercy-admit] all admitted (None/Low)"
  exit 0
else
  status=$?
  if [[ "${WHITEHAT_ALLOW_MEDIUM:-0}" == "1" ]]; then
    echo "[mercy-admit] WHITEHAT_ALLOW_MEDIUM=1 set — allowing despite gate (human-reviewed path only)"
    exit 0
  fi
  echo "[mercy-admit] BLOCKED — Medium+ content detected. Fix or use human-reviewed escape hatch."
  exit "$status"
fi
