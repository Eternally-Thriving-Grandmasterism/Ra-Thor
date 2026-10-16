# Bounded Cursor handoff protocol

**Date:** 2026-10-10  
**Status:** Draft — human review required. Not a product. Not a permanent delegation.  
**Authority:** PATSAGi minute `2026-10-10-Seat-Vs-Cursor-Bounded` under TOLC 8  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6 (unchanged)  
**Related:** [`docs/science/PATSAGI-COUNCIL-MINUTE-2026-10-10-SEAT-VS-CURSOR.md`](PATSAGI-COUNCIL-MINUTE-2026-10-10-SEAT-VS-CURSOR.md), root `PUBLIC_CLAIM.lock.md`

This protocol describes how an external agent surface (Cursor running Grok 4.7 high non-fast, or equivalent) may be used for mechanical tasks without replacing the primary seat.

---

## 1. Primary seat (non-delegable)

The following surfaces stay on this seat (current operator-instructed Grok session with direct GitHub tools) unless a later named PATSAGi motion says otherwise:

- PATSAGi council minutes and resolution records
- `PUBLIC_CLAIM.lock.md` and any public-claim language
- Root `Cargo.toml` `[workspace].members` and `TIER_MAP.md` membership decisions
- Any change that would expand or weaken the claim lock, Core Tier-1 gate, or human-review rule
- Final merge decision on any PR that touches the above

## 2. Allowed bounded handoff

Cursor (or equivalent) may be used for mechanical tasks such as:

- Expanding the on-disk research sketch at `crates/legal-research-source-locked/` (citation helpers, fixtures, tests) without adding it to default members
- Generating or refining test fixtures from public source text
- Running focused `cargo test -p` or `cargo check` on Core crates and reporting results
- Drafting non-claim documentation that carries the draft seal
- Mechanical refactors that do not alter public claims or membership

## 3. Mandatory constraints

Every Cursor session under this protocol must:

1. Receive an explicit human instruction naming the bounded task and the surfaces it may touch.
2. Produce only drafts. Every public or committed artifact must carry a clear draft seal (example: “DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.”).
3. Never modify `PUBLIC_CLAIM.lock.md`, root `Cargo.toml` members, or `TIER_MAP.md` without a prior named motion and human confirmation.
4. Never claim product status, certification, legal resolution, or autonomous shipping.
5. Leave a reviewable diff. The human reviews and approves before merge.
6. Prefer single-path, path-filtered reads. Do not recursive-walk the repository root.
7. Stay inside the current workspace identity (14.15.6) and the existing AG-SML v1.1 grant language.

## 4. Handoff steps

1. Human writes a short task card naming: goal, allowed paths, forbidden paths, and expected output form.
2. Cursor session executes only within those bounds and produces a branch or patch.
3. Human reviews the diff against the constraints above.
4. If acceptable, human merges (or instructs this seat to merge). If not, the patch is discarded or revised.
5. Any expansion beyond the original task card requires a new card or a named PATSAGi motion.

## 5. Rejection cases

Immediately stop and return to this seat if the agent:

- Attempts to edit the claim lock, membership, or minutes without explicit prior approval
- Produces language that overclaims (legal resolver, certified product, AGSi warranty, etc.)
- Bypasses human review
- Requests permanent or open-ended delegation

## 6. Status of this document

This file is a draft protocol. It does not authorize unrestricted use. It does not change default workspace members. It does not alter the public claim lock. A human may redline it. Contact remains info@Rathor.ai.

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡
