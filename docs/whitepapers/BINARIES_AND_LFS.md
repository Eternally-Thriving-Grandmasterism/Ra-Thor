# Git LFS Guidance for Ra-Thor Whitepaper Assets (v2.1)

## Current Status on Branch

Text assets, Lean formalization (Phases 1-5 + FFI), Esacheck examples (14+), Creusot/Prusti/Viper sketches, and test crate are complete and pushed.

**Remaining large binaries prepared locally:**
- `Ra-Thor-Whitepaper-v2.1-Visual-Companion-Forensic.pdf` (≈1.2 MB, Platypus clean, all 4 diagrams + forensic framing)
- `Ra-Thor-Executive-Summary-One-Page-v2.0.pdf` (one-page pitch, zero overlaps)
- 4 Architecture Diagrams (high-resolution JPGs)

## Recommended Workflow: Git LFS (Best Practice)

```bash
# In your local clone of Eternally-Thriving-Grandmasterism/Ra-Thor

git checkout feature/whitepaper-v2.1-platypus-clean-forensic

git lfs install

git lfs track "*.pdf"
git lfs track "docs/diagrams/*.jpg"
git add .gitattributes

git commit -m "chore: Track PDFs and diagrams with Git LFS"

# Copy the prepared files from artifacts
cp /home/workdir/artifacts/whitepapers/Ra-Thor-Whitepaper-v2.1-Visual-Companion-Forensic.pdf docs/whitepapers/
cp /home/workdir/artifacts/whitepapers/Ra-Thor-Executive-Summary-One-Page-v2.0.pdf docs/whitepapers/

mkdir -p docs/diagrams
cp /home/workdir/artifacts/whitepapers/diagrams/*.jpg docs/diagrams/

git add docs/whitepapers/*.pdf docs/diagrams/*.jpg .gitattributes

git commit -m "docs(assets): Add Visual Companion PDF, Executive Summary, and all 4 architecture diagrams via LFS"

git push origin feature/whitepaper-v2.1-platypus-clean-forensic
```

## Alternative (Simple Local Commit)

If you prefer not to use LFS immediately:
```bash
# From the branch
cp /home/workdir/artifacts/whitepapers/*.pdf docs/whitepapers/
cp /home/workdir/artifacts/whitepapers/diagrams/*.jpg docs/diagrams/
git add docs/whitepapers/*.pdf docs/diagrams/*.jpg
git commit -m "docs(assets): Add remaining PDFs and diagrams"
git push
```

## After Adding

- The PDFs will be downloadable from the repo and GitHub releases.
- Diagrams will render nicely in the whitepaper Markdown and Visual Companion.
- This completes the professional evidence package for PR #159.

**One Organism. Mercy First. Truth Forensically Distilled.**