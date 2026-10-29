#!/bin/bash
# Ra-Thor Professional Assets LFS Commit Script
# Run this from the root of your Ra-Thor clone after pulling latest main

set -e

echo "=== Ra-Thor Professional PDFs & Diagrams LFS Commit ==="

# Ensure LFS is installed and tracking is set
git lfs install

echo "Copying professional assets from artifacts..."

mkdir -p docs/whitepapers docs/diagrams

# Copy PDFs
cp /home/workdir/artifacts/whitepapers/Ra-Thor-Executive-Summary-One-Page-v2.0.pdf docs/whitepapers/ 2>/dev/null || echo "Executive Summary PDF not found in artifacts"
cp /home/workdir/artifacts/whitepapers/Ra-Thor-Whitepaper-v2.0-Visual-Companion-Forensic.pdf docs/whitepapers/ 2>/dev/null || echo "Visual Companion PDF not found in artifacts"

# Copy diagrams
cp /home/workdir/artifacts/whitepapers/diagrams/*.jpg docs/diagrams/ 2>/dev/null || echo "Diagrams not found in artifacts"

# Add via LFS
git add docs/whitepapers/*.pdf docs/diagrams/*.jpg .gitattributes

git commit -m "docs(assets): Add professional PDFs and architecture diagrams via LFS

- Executive Summary (one-page, v2.0)
- Visual Companion PDF (forensic)
- All 4 architecture diagrams (TOLC 8, PATSAGi One Organism, Self-Evolution, Full System)

One Organism. Mercy First. Truth Forensically Distilled."

echo "Done. Now run: git push origin main"
