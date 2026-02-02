#!/usr/bin/env bash
# clean-logs.sh – Purge entropy artifacts

find . -type f \( -name "*.log" -o -name "*.tmp" -o -name "*.bak" \) -delete
git clean -fdx --exclude=.gitignore --dry-run

echo "Mercy entropy purge complete"
