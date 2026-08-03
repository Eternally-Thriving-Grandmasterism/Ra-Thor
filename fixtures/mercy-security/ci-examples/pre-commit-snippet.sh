#!/usr/bin/env bash
# Minimal pre-commit example that scans the public fixture corpus (or staged files).
#
# Install as git hook (from monorepo root):
#   ln -sf ../../fixtures/mercy-security/ci-examples/pre-commit-snippet.sh .git/hooks/pre-commit
#
# Or call directly for a quick smoke test against the public corpus.
#
# Escape hatch (human-reviewed only):
#   WHITEHAT_ALLOW_MEDIUM=1 ./fixtures/mercy-security/ci-examples/pre-commit-snippet.sh
#
# Contact: info@Rathor.ai | White-hat only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BIN="target/debug/mercy-admit"
if [[ ! -x "$BIN" ]]; then
  echo "[mercy-admit] building binary..."
  cargo build -q -p mercy-security --bin mercy-admit
fi

# Default: exercise the public corpus (benign must pass, blocked must fail)
FILES=(
  fixtures/mercy-security/benign/model_card_clean.md
  fixtures/mercy-security/blocked/trust_remote_code_loader.txt
)

echo "[mercy-admit] scanning public fixture corpus samples..."

set +e
"$BIN" --verbose "${FILES[@]}"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  echo "[mercy-admit] unexpected: blocked fixture was admitted"
  exit 1
fi

if [[ "${WHITEHAT_ALLOW_MEDIUM:-0}" == "1" ]]; then
  echo "[mercy-admit] WHITEHAT_ALLOW_MEDIUM=1 — allowing (human-reviewed path only)"
  exit 0
fi

echo "[mercy-admit] gate correctly tripped on blocked fixture (exit $status)"
exit 0
