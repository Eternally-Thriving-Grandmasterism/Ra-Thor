# legal-research-source-locked

**Status:** On-disk research sketch only. **Not** a workspace member. **Not** a legal product. **Not** a resolver.

**Authority:** PATSAGi minute `2026-10-10-Legal-Research-Bound-NotResolver` under TOLC 8.  
**Contact:** info@Rathor.ai  
**Lock:** See root `PUBLIC_CLAIM.lock.md`. Outputs are drafts. A human must review them before filing, sale, or public legal claims.

## Purpose

Inspectable research tooling for public legal source material:

- Source-locked parsing of public judgments and statutes (no inference beyond the supplied text).
- Citation extraction and parallel-citation helpers.
- Structured head-of-power / division-of-powers mapping (Constitution Act, 1867 ss. 91 / 92 / 92A etc.) as research notes.
- Draft structured analyses that carry the mandatory “drafts — human review required” seal.

This is research software for operators who already possess the source text. It does not provide legal advice, resolution, or opinions. It does not hold itself out as a lawyer, law firm, or certified product.

## Boundaries

- Not added to root `Cargo.toml` `[workspace].members`.
- Does not expand Core Tier-1.
- Does not weaken the public claim lock.
- Human remains the reviewer and the signer.
- Example surface (e.g. 2026 ABCA 320) may be analyzed under the current research regime; it does not justify a product change.

## Next honest work

Keep Core Tier-1 green. Any expansion of this sketch requires a later named PATSAGi motion.

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡

## Citation scan added 2026-10-10

Task card: `docs/science/TASK-CARD-CURSOR-2026-10-10-LEGAL-RESEARCH-STUB.md`.

`extract_citations_research` copies spans that match `\d{4}\s+[A-Z]+\s+\d+` (token boundaries on the year and the number) into a `ResearchDraft`. `find_reporter_citations` returns those same spans. The draft body lists the verbatim spans from the supplied text.

Fixture `fixtures/synthetic-citation-list.txt` is a synthetic stand-in, not a judgment. `tests/extract_reporter_citations.rs` checks the span list and the draft seal.

Volume-reporter-page forms such as `410 U.S. 113` stay outside this pattern. Coverage of those forms is not demonstrated.

This crate stays off default workspace members. Inside this repository, `cargo test --manifest-path crates/legal-research-source-locked/Cargo.toml` stops because Cargo treats the path as a non-member. The same tests passed on 2026-10-10 when the crate directory was copied outside the workspace and `cargo test --offline` was run there.

DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.
