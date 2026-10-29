#!/usr/bin/env bash
# Promote modular shell to live index.html (after i18n modules are on disk).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "$ROOT/index.modular.html" ]]; then
  echo "Missing index.modular.html at repo root."
  exit 1
fi
if [[ ! -d "$ROOT/i18n" ]]; then
  echo "Missing i18n/ directory."
  exit 1
fi
cp "$ROOT/index.html" "$ROOT/index.html.pre-modular.bak"
cp "$ROOT/index.modular.html" "$ROOT/index.html"
echo "OK: index.html is now the modular shell. Backup: index.html.pre-modular.bak"
echo "Commit when ready. Contact: info@Rathor.ai"
