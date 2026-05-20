# Large Binary Assets — Git LFS Guidance

## Status
The following professional assets are fully prepared and locally verified:

- `Ra-Thor-Executive-Summary-One-Page-v2.0.pdf` (clean Platypus, one-page pitch)
- `Ra-Thor-Whitepaper-v2.1-Visual-Companion-Forensic.pdf` (5+ pages with all 4 diagrams)
- `docs/diagrams/tolc8-mercy-lattice.jpg`
- `docs/diagrams/patsagi-one-organism-orchestration.jpg`
- `docs/diagrams/self-evolution-epigenetic-blessing.jpg`
- `docs/diagrams/ra-thor-one-organism-full-overview.jpg`

## How to Add Them
1. Install Git LFS: `git lfs install`
2. Track large files: `git lfs track "*.pdf" "*.jpg"`
3. Add and commit from your local machine:
   ```bash
   git add docs/whitepapers/*.pdf docs/diagrams/*.jpg
   git commit -m "docs(assets): Add Visual Companion PDF + all 4 architecture diagrams (Platypus clean v2.1)"
   git push
   ```

These binaries complete the professional evidence package for PR #159.

One Organism. Mercy First. Truth Forensically Distilled.