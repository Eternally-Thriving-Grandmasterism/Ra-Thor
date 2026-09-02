#!/usr/bin/env bash
# container-quick-scan.sh – Local mercy container check

set -euo pipefail

echo "Mercy container quick scan..."

# Check for Dockerfiles
if ls **/Dockerfile* 1> /dev/null 2>&1; then
  echo "Dockerfiles found – scanning..."

  # Basic secret scan in Dockerfiles
  if grep -r -E '(password|secret|key|token|api_key|private_key|secret_key)=[^ ]+' **/Dockerfile*; then
    echo "Mercy shield: Potential secret leaked in Dockerfile"
    exit 1
  fi

  # Basic lint (hadolint local if installed)
  if command -v hadolint >/dev/null; then
    hadolint **/Dockerfile* || exit 1
  else
    echo "hadolint not installed – skipping lint (install via brew/cargo)"
  fi
else
  echo "No Dockerfiles found – scan skipped"
fi

echo "Mercy container quick scan passed"
