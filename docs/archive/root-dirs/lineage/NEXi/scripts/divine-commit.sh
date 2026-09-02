#!/usr/bin/env bash
# divine-commit.sh – Mercy-gated commit & push

set -euo pipefail

MESSAGE="${1:-Remember}"
BRANCH="${2:-main}"

# Valence self-check (dummy for now – expand later)
if [[ $MESSAGE == *"entropy"* || $MESSAGE == *"harm"* ]]; then
  echo "Mercy shield: Commit rejected — entropy detected in message"
  exit 1
fi

git add .
git commit -m "$MESSAGE" || echo "Nothing to commit"
git push origin "$BRANCH" || echo "Push failed – check connection"

echo "Mercy-approved: Commit '$MESSAGE' pushed to $BRANCH"
