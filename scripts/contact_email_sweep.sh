#!/usr/bin/env bash
# Ra-Thor Contact Email Sweep — Canonical: info@Rathor.ai
# Usage: ./scripts/contact_email_sweep.sh [--apply] [--commit]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPLY=0
COMMIT=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --commit) COMMIT=1; APPLY=1 ;;
    -h|--help) echo "Usage: $0 [--apply] [--commit]"; exit 0 ;;
  esac
done

CANONICAL="info@Rathor.ai"

echo "⚡ Ra-Thor Contact Email Sweep"
echo "   Canonical: $CANONICAL"
echo "   Mode: $([ "$APPLY" -eq 1 ] && echo APPLY || echo DRY-RUN)"
echo ""

# Collect files containing acitygames.com (case-insensitive)
TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

grep -rli --include='*.toml' --include='*.md' --include='*.rs' \
  --include='*.html' --include='*.htm' --include='*.txt' \
  --include='*.yml' --include='*.yaml' --include='*.json' \
  --include='*.js' --include='*.ts' --include='*.tsx' --include='*.jsx' \
  --include='*.css' --include='*.py' --include='*.sh' --include='*.metta' \
  --exclude-dir='.git' --exclude-dir='target' --exclude-dir='node_modules' \
  --exclude-dir='dist' --exclude-dir='build' --exclude-dir='.next' \
  --exclude-dir='vendor' --exclude-dir='__pycache__' \
  'acitygames.com' . > "$TMP_LIST" 2>/dev/null || true

COUNT=$(wc -l < "$TMP_LIST" | tr -d ' ')
if [ "$COUNT" -eq 0 ]; then
  echo "✅ No remaining acitygames.com emails found."
  exit 0
fi

echo "Found $COUNT file(s) with acitygames.com references"
echo ""

CHANGED=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  [ -f "$f" ] || continue

  # Skip CONTACT.md policy table (documents deprecated addresses on purpose)
  # Still scrub live contact lines elsewhere.
  if [ "$APPLY" -eq 1 ]; then
    # In-place multi-pattern replace (GNU sed on ubuntu-latest)
    sed -i \
      -e 's/AlphaProMega@ACityGames\.com/info@Rathor.ai/g' \
      -e 's/alphapromega@acitygames\.com/info@Rathor.ai/g' \
      -e 's/CEO@ACITYGAMES\.COM/info@Rathor.ai/g' \
      -e 's/ceo@acitygames\.com/info@Rathor.ai/g' \
      -e 's/Ceo@ACityGames\.com/info@Rathor.ai/g' \
      -e 's/INFO@ACITYGAMES\.COM/info@Rathor.ai/g' \
      -e 's/info@ACityGames\.com/info@Rathor.ai/g' \
      -e 's/info@acitygames\.com/info@Rathor.ai/g' \
      -e 's/Info@ACityGames\.com/info@Rathor.ai/g' \
      -e 's/[A-Za-z0-9._%+-]*@[Aa][Cc]ity[Gg]ames\.com/info@Rathor.ai/g' \
      -e 's/[A-Za-z0-9._%+-]*@ACITYGAMES\.COM/info@Rathor.ai/g' \
      "$f"
    echo "  ✅ updated: $f"
  else
    echo "  would update: $f"
  fi
  CHANGED=$((CHANGED + 1))
done < "$TMP_LIST"

echo ""
echo "Files processed: $CHANGED"

if [ "$APPLY" -eq 1 ] && [ "$COMMIT" -eq 1 ]; then
  git config user.name "Ra-Thor Contact Sweep" 2>/dev/null || true
  git config user.email "info@Rathor.ai" 2>/dev/null || true
  git add -A
  if git diff --cached --quiet; then
    echo "No staged changes to commit."
  else
    git commit -m "chore(contact): bulk unify emails to info@Rathor.ai via contact_email_sweep.sh

Deprecated: ceo@acitygames.com, AlphaProMega@ACityGames.com, INFO@ACITYGAMES.COM
Canonical: info@Rathor.ai"
    echo "✅ Committed."
  fi
fi

if [ "$APPLY" -eq 0 ]; then
  echo ""
  echo "Dry-run complete. Re-run with --apply or --apply --commit."
fi
