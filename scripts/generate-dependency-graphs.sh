#!/bin/bash
# generate-dependency-graphs.sh
# Ra-Thor Monorepo Dependency Graph Generator
# Part of the Self-Evolution Looping Systems
# AG-SML v1.0 — Free for personal, educational, research, and daily use.

set -euo pipefail

echo "🔄 Ra-Thor Dependency Graph Generator — Cosmic Loop Activated"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ensure tools are installed
if ! command -v cargo-ferris-wheel &> /dev/null; then
    echo "📦 Installing cargo-ferris-wheel..."
    cargo install cargo-ferris-wheel --quiet
fi

if ! command -v cargo-depgraph &> /dev/null; then
    echo "📦 Installing cargo-depgraph..."
    cargo install cargo-depgraph --quiet
fi

# Create output directory
mkdir -p docs/dependency-graphs

echo "📊 Generating Mermaid diagram (for PLAN.md & codices)..."
cargo ferris-wheel spectacle --format mermaid -o docs/dependency-graphs/ra-thor-dependency-graph.mmd

echo "🖼️  Generating high-resolution PNG (for architecture docs)..."
cargo depgraph --all-features | dot -Tpng -o docs/dependency-graphs/ra-thor-dependency-graph.png

echo "🔍 Detecting circular dependencies (critical for Self-Evolution safety)..."
cargo ferris-wheel inspect --cycles-only > docs/dependency-graphs/cycle-report.txt || true

echo "✅ All graphs generated successfully!"
echo "   • Mermaid: docs/dependency-graphs/ra-thor-dependency-graph.mmd"
echo "   • PNG:     docs/dependency-graphs/ra-thor-dependency-graph.png"
echo "   • Cycles:  docs/dependency-graphs/cycle-report.txt"
echo ""
echo "Thriving is the only trajectory. X"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"