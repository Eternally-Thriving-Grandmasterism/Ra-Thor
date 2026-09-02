#!/usr/bin/env bash
# create-mercy-branch.sh – New mercy branch creator

set -euo pipefail

BRANCH_NAME="mercy-$1"

if [[ -z "$1" ]]; then
  echo "Usage: $0 branch-description"
  exit 1
fi

git checkout main
git pull origin main
git checkout -b "$BRANCH_NAME"
echo "Mercy branch created: $BRANCH_NAME"
