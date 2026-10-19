# GROK BOT → CURSOR HANDOFF CARD — index.html Card & FAQ Update v1.0

**Status:** Operational handoff note for Grok Bots | Under TOLC 8  
**Date:** 2026-10-19  
**Contact:** info@Rathor.ai  
**License:** AG-SML  
**Purpose:** Direct Grok Bots to engage a Cursor agent for a precise, non-destructive edit of the live `index.html` on the Rathor.ai site.

---

## Task for Cursor Agent

Update the Ra-Thor on X card subtitle and the related Architecture FAQ answers in `index.html` (and any matching i18n strings) to reflect current Grok-on-X capabilities.

### Required change (card subtitle)

**Find:**  
`A lighter, less agentic Grok surface in the X app or at x.com. Optional.`

**Replace with:**  
`A Grok-on-X session in the X app or at x.com. Now includes skills, connectors, and file upload, with stronger performance. Optional.`

### FAQ answers (if present as static text or i18n)

Use the recommended wording already recorded in:  
`docs/FAQ_GROK_X_BUTTON_UPDATE_NOTE_2026-10-18.md`

- Keep all other content, structure, assets, and scripts untouched.  
- Do not truncate or rewrite the file.  
- Preserve the independent posture (no affiliation claims, drafts only, Grok remains an xAI product).

## Constraints for the Cursor agent

- Edit only the identified subtitle line and the matching FAQ answer text.  
- Verify the change with a diff before commit.  
- Commit message: `Update Ra-Thor on X card subtitle and FAQ for Grok-on-X skills/connectors`  
- Human review recommended before the live site is considered final.

## Bot instructions

Grok Bots: load this card on rewake when the index.html update is pending.  
Locate a Cursor agent (or equivalent coding agent) that can safely edit large HTML files.  
Hand off the exact find/replace above plus the FAQ note path.  
Do not attempt the full-file rewrite from this seat.

**Thunder locked. ONE Organism. Yoi ⚡**
