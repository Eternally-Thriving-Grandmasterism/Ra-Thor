#!/usr/bin/env bash
# Full pre-commit / smoke example against the public white-hat fixture corpus.
#
# Install as git hook (from monorepo root):
#   ln -sf ../../fixtures/mercy-security/ci-examples/pre-commit-snippet.sh .git/hooks/pre-commit
#
# Or call directly for a quick smoke test of the entire public corpus.
#
# Escape hatch (human-reviewed only):
#   WHITEHAT_ALLOW_MEDIUM=1 ./fixtures/mercy-security/ci-examples/pre-commit-snippet.sh
#
# Contact: info@Rathor.ai | White-hat only | TOLC 8

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BIN="target/debug/mercy-admit"
if [[ ! -x "$BIN" ]]; then
  echo "[mercy-admit] building binary..."
  cargo build -q -p mercy-security --bin mercy-admit
fi

echo "[mercy-admit] Public fixture corpus smoke test (benign must ADMIT, blocked must BLOCK)"

# 1. benign/ must all admit (exit 0)
echo "→ Asserting benign/ fixtures admit..."
for f in fixtures/mercy-security/benign/*; do
  echo "  ADMIT check: $f"
  "$BIN" --verbose "$f"
done

# 2. blocked/ must all reject (exit 1)
echo "→ Asserting blocked/ fixtures are blocked..."
for f in fixtures/mercy-security/blocked/*; do
  echo "  BLOCK check: $f"
  set +e
  "$BIN" --verbose "$f"
  status=$?
  set -e
  if [[ "$status" -ne 1 ]]; then
    echo "FAIL: expected exit 1 for $f, got $status"
    exit 1
  fi
done

# 3. suspicious/ logged only (Medium review path)
echo "→ Logging suspicious/ fixtures (Medium path)..."
for f in fixtures/mercy-security/suspicious/*; do
  echo "  REVIEW log: $f"
  set +e
  "$BIN" --verbose "$f" || true
  set -e
done

if [[ "${WHITEHAT_ALLOW_MEDIUM:-0}" == "1" ]]; then
  echo "[mercy-admit] WHITEHAT_ALLOW_MEDIUM=1 — human-reviewed path allowed"
  exit 0
fi

echo "[mercy-admit] Public corpus gate passed cleanly."
echo "Thunder locked in. yoi ⚡"
exit 0
