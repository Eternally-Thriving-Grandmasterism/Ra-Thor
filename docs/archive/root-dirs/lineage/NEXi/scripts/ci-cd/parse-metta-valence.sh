#!/usr/bin/env bash
# parse-metta-valence.sh – Ensure code respects MeTTa mercy rules

set -euo pipefail

MEETA_FILE="docs/mercy_core_atoms.metta"
CODE_DIRS="mercy_* core"

echo "Mercy MeTTa valence validation..."

# Check if threshold is respected in code
if ! grep -r "0.9999999" $CODE_DIRS; then
  echo "Mercy shield: Valence threshold missing in code"
  exit 1
fi

# Basic syntax check on .metta files
if ! grep -q "(= (valence-threshold) 0.9999999)" "$MEETA_FILE"; then
  echo "Mercy shield: Core valence threshold missing in MeTTa"
  exit 1
fi

echo "Mercy MeTTa valence validation passed"
