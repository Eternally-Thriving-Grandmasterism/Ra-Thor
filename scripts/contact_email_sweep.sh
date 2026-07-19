#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════
# Ra-Thor Contact Email Sweep — Canonical: info@Rathor.ai
# AG-SML v1.0 | 2026-07-19
#
# Usage:
#   ./scripts/contact_email_sweep.sh              # dry-run (list matches)
#   ./scripts/contact_email_sweep.sh --apply      # write changes
#   ./scripts/contact_email_sweep.sh --apply --commit  # write + git commit
# ═════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPLY=0
COMMIT=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --commit) COMMIT=1; APPLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
  esac
done

CANONICAL="info@Rathor.ai"

# Deprecated patterns → canonical (order matters: longer / more specific first)
declare -a PATTERNS=(
  "AlphaProMega@ACityGames.com"
  "alphapromega@acitygames.com"
  "CEO@ACITYGAMES.COM"
  "ceo@acitygames.com"
  "Ceo@ACityGames.com"
  "INFO@ACITYGAMES.COM"
  "info@ACityGames.com"
  "info@acitygames.com"
  "Info@ACityGames.com"
)

# File types to touch (text sources only)
EXTENSIONS=(
  "toml" "md" "rs" "html" "htm" "txt" "yml" "yaml" "json"
  "js" "ts" "tsx" "jsx" "css" "scss" "py" "sh" "metta"
)

# Build find predicates
FIND_ARGS=()
for i in "${!EXTENSIONS[@]}"; do
  if [[ $i -eq 0 ]]; then
    FIND_ARGS+=( -name "*.${EXTENSIONS[$i]}" )
  else
    FIND_ARGS+=( -o -name "*.${EXTENSIONS[$i]}" )
  fi
done

# Exclude binary-ish / generated dirs
EXCLUDE_DIRS=(
  ".git" "target" "node_modules" "dist" "build" ".next"
  "vendor" "__pycache__" ".cargo"
)

PRUNE_ARGS=()
for d in "${EXCLUDE_DIRS[@]}"; do
  PRUNE_ARGS+=( -path "./$d" -o -path "*/$d/*" -o )
done
# trailing false so last -o is valid
PRUNE_EXPR=( \( "${PRUNE_ARGS[@]}" -false \) -prune -o )

echo "⚡ Ra-Thor Contact Email Sweep"
echo "   Canonical: $CANONICAL"
echo "   Mode: $([ "$APPLY" -eq 1 ] && echo APPLY || echo DRY-RUN)"
echo ""

# Collect candidate files that contain any deprecated pattern
mapfile -t CANDIDATES < <(
  find . "${PRUNE_EXPR[@]}" -type f \( "${FIND_ARGS[@]}" \) -print0 \
    | xargs -0 grep -l -i -E 'acitygames\.com' 2>/dev/null || true
)

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "✅ No remaining deprecated acitygames.com emails found."
  exit 0
fi

echo "Found ${#CANDIDATES[@]} file(s) with acitygames.com references:"
printf '  %s\n' "${CANDIDATES[@]}"
echo ""

CHANGED=0
for f in "${CANDIDATES[@]}"; do
  before=$(wc -c < "$f" | tr -d ' ')
  content=$(cat "$f")
  original="$content"

  for p in "${PATTERNS[@]}"; do
    # case-sensitive exact replacements for known forms
    content="${content//$p/$CANONICAL}"
  done

  # Catch any remaining *@acitygames.com / *@ACityGames.com (case variants)
  # via sed for residual patterns
  if [[ "$content" != "$original" ]]; then
    if [[ "$APPLY" -eq 1 ]]; then
      printf '%s' "$content" > "$f"
      echo "  ✅ updated: $f"
    else
      echo "  would update: $f"
    fi
    CHANGED=$((CHANGED + 1))
  fi
done

# Residual catch-all with sed (case-insensitive email local-part @ domain)
if [[ "$APPLY" -eq 1 ]]; then
  for f in "${CANDIDATES[@]}"; do
    # Only touch if still has acitygames.com
    if grep -qi 'acitygames\.com' "$f" 2>/dev/null; then
      # GNU/BSD sed compatible: replace remaining *@*acitygames.com
      if sed --version >/dev/null 2>&1; then
        # GNU sed
        sed -i -E \
          -e 's/[A-Za-z0-9._%+-]+@[Aa][Cc]ity[Gg]ames\.com/info@Rathor.ai/g' \
          -e 's/[A-Za-z0-9._%+-]+@ACITYGAMES\.COM/info@Rathor.ai/g' \
          "$f"
      else
        # BSD sed (macOS)
        sed -i '' -E \
          -e 's/[A-Za-z0-9._%+-]+@[Aa][Cc]ity[Gg]ames\.com/info@Rathor.ai/g' \
          -e 's/[A-Za-z0-9._%+-]+@ACITYGAMES\.COM/info@Rathor.ai/g' \
          "$f"
      fi
      echo "  ✅ residual scrub: $f"
    fi
  done
fi

echo ""
echo "Files touched (or would touch): $CHANGED"

if [[ "$APPLY" -eq 1 && "$COMMIT" -eq 1 ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git add -A
    if git diff --cached --quiet; then
      echo "No staged changes to commit."
    else
      git commit -m "chore(contact): bulk unify emails to info@Rathor.ai via contact_email_sweep.sh

Deprecated: ceo@acitygames.com, AlphaProMega@ACityGames.com, INFO@ACITYGAMES.COM
Canonical: info@Rathor.ai
Policy: CONTACT.md"
      echo "✅ Committed."
    fi
  else
    echo "Not a git repo — skip commit."
  fi
fi

if [[ "$APPLY" -eq 0 ]]; then
  echo ""
  echo "Dry-run complete. Re-run with --apply to write, or --apply --commit to write+commit."
fi
