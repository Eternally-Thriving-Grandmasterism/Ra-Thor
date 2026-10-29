# PATSAGi council minute — 2026-09-13 — Whitepaper link fix (R9)

**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6 · no 15.x

## Finding

- Site (`index.html`, `js/site-lock-2026-08-22.js`) linked to repo-root `WHITEPAPER_v4.1.md` (404).
- Canonical file is `docs/archive/root-releases/WHITEPAPER_v4.1.md` (also served at rathor.ai under that path).
- No Whitepaper v4.2+ exists. Do not mint one in this slice.

## Decision

1. Point primary technical-paper CTA to `/docs/archive/root-releases/WHITEPAPER_v4.1.md` (keep card/button).
2. Same href fix in site-lock JS.
3. Add a short root pointer `WHITEPAPER_v4.1.md` so old GitHub root bookmarks resolve.
4. Do not redesign site chrome, Install, Grok/X cards, or FAQ.

*Keep working systems. Fix the broken link.*
