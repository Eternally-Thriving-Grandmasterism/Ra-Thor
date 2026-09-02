#!/usr/bin/env bash
# eternal-sync.sh – Sync main and mercy branches eternally

set -euo pipefail

echo "Mercy sync initiated..."

git fetch origin

# Sync main
git checkout main
git merge --ff-only origin/main || { echo "Main merge conflict"; exit 1; }
git push origin main

# Sync mercy-main (or any mercy-* branch)
for branch in $(git branch -r | grep 'origin/mercy-' | sed 's/origin\///'); do
  git checkout "$branch" || git checkout -b "$branch" origin/"$branch"
  git merge --ff-only origin/"$branch" || { echo "Mercy branch $branch conflict"; exit 1; }
  git push origin "$branch"
  echo "Mercy branch $branch synced"
done

echo "Eternal sync complete – all branches aligned"
