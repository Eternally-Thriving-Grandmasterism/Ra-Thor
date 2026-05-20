#!/bin/bash
# Ra-Thor Professional Asset Integration Script
# Run from your Ra-Thor clone root

echo "Ra-Thor: Integrating professional PDFs + Diagrams via Git LFS"

git lfs install
git lfs track "*.pdf" "*.jpg"
git add .gitattributes

mkdir -p docs/whitepapers docs/diagrams

# Adjust source path as needed
cp ~/artifacts/whitepapers/*.pdf docs/whitepapers/ 2>/dev/null || true
cp ~/artifacts/whitepapers/diagrams/*.jpg docs/diagrams/ 2>/dev/null || true

git add docs/whitepapers/*.pdf docs/diagrams/*.jpg
git commit -m "docs(assets): Add professional PDFs and architecture diagrams via LFS"
git push origin main

echo "Done. Professional assets integrated."