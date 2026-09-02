#!/usr/bin/env bash
# security-quick-scan.sh – Local mercy security check

set -euo pipefail

echo "Mercy quick security scan..."

# Check for secrets (simple grep – expand with trufflehog local later)
if grep -r -E '(password|secret|key|token|api_key|private_key|secret_key)=[^ ]+' . --exclude-dir={.git,node_modules,target,venv}; then
  echo "Mercy shield: Potential secret leaked in code"
  exit 1
fi

# Check Cargo.toml for untrusted deps (manual list – expand later)
if grep -q "unsafe" Cargo.toml; then
  echo "Mercy caution: 'unsafe' keyword detected – review required"
fi

echo "Mercy quick scan passed – safe to commit"
