#!/usr/bin/env bash
# dependency-audit.sh – Local vuln scan

cargo install cargo-audit --locked || echo "cargo-audit already installed"
cargo audit --ignore RUSTSEC-202X-XXXX  # add known false positives if any

echo "Mercy dependency audit complete"
