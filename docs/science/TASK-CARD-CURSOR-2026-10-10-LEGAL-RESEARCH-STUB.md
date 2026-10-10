# Task card — Cursor bounded handoff

**Date:** 2026-10-10  
**Status:** Draft task card — human review required before use.  
**Protocol:** [`docs/science/BOUNDED-CURSOR-HANDOFF-PROTOCOL-2026-10-10.md`](BOUNDED-CURSOR-HANDOFF-PROTOCOL-2026-10-10.md)  
**Authority:** PATSAGi minute `2026-10-10-Seat-Vs-Cursor-Bounded`  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6 (unchanged)

---

## Goal

Expand the on-disk research sketch at `crates/legal-research-source-locked/` with a minimal, source-locked citation extractor and one public-text fixture. Produce only drafts. Do not claim product status or legal resolution.

## Allowed paths

- `crates/legal-research-source-locked/src/` (lib.rs and any new modules under it)
- `crates/legal-research-source-locked/tests/` (new test files only)
- `crates/legal-research-source-locked/fixtures/` (new public-text fixtures only)
- `crates/legal-research-source-locked/README.md` (append-only notes about what was added; keep the existing boundaries)
- `crates/legal-research-source-locked/Cargo.toml` (dev-dependencies only if required for the tests; no version or publish changes)

## Forbidden paths

- Root `Cargo.toml` (do not add the crate to `[workspace].members`)
- `TIER_MAP.md`
- `PUBLIC_CLAIM.lock.md`
- Any file under `docs/science/` except this task card itself if a status note is required
- Any other crate
- Any workflow, CI, or security file

## Expected output

1. A small function in the crate that extracts simple citation patterns (e.g. `\d{4}\s+[A-Z]+\s+\d+` style Canadian/US reporter patterns) from a supplied string and returns a `ResearchDraft` carrying the mandatory seal.
2. One fixture file containing a short public excerpt (or synthetic stand-in) of a court citation list.
3. One unit or integration test that runs the extractor against the fixture and asserts the draft seal is present.
4. A short README note stating what was added and repeating: “DRAFT — human review required. Not legal advice. Not a product.”
5. A branch or patch ready for human review. No direct push to main.

## Constraints (must carry)

- Every committed or public artifact must include or reference the seal:  
  `DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.`
- Do not modify default workspace members.
- Do not recursive-walk the repository root.
- Do not claim the crate is a legal resolver, product, or certified tool.
- Human reviews the full diff before any merge.

## Rejection triggers

Return the work to the primary seat immediately if the session attempts to touch forbidden paths, weaken the seal, or produce overclaiming language.

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡
