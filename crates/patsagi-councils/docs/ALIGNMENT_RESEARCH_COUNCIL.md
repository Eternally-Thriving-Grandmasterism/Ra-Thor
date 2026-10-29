# AlignmentResearchCouncil — PATSAGi sandbox (forest name)

**Workspace identity:** 14.15.6  
**Contact:** info@Rathor.ai  
**Status:** inspectable research software. Forest name only in this crate. Lived adapter: `lattice-conductor-v14::alignment_research`.

This file names a sandbox council under the PATSAGi forest. It does **not** add `patsagi-councils` to `[workspace].members`. It does **not** claim a running alignment researcher.

Capable · Bounded · Corrigible.

---

## What this council is

`AlignmentResearchCouncil` is a gated proposal sandbox. Parallel researcher stubs may emit alignment tests, SAE probes, or gate refinements as **proposals**.

Lived types and named tests:

- `crates/lattice-conductor-v14/src/alignment_research.rs`
- Fixture: `crates/lattice-conductor-v14/fixtures/alignment_research_proposal_audit_v0.json`
- Verify: `cargo test -p lattice-conductor-v14 alignment_research`

## Law (verbatim with the R&D brief)

- TOLC 8 Layer 0 is non-bypassable. The sandbox sits **under** the gates.
- Valence floor stays at or above 0.999999. The researcher cannot lower it.
- Council 13 / human override remains. The researcher cannot merge `main`.
- Every artifact attaches an Opportunity 3 evidence row.
- Audit actor ≠ researcher actor. The researcher process cannot write the score.
- Promote / merge / threshold-edit Reject without Council 13 or the existing override path. Merge and Layer 0 writes Reject even then.

## Not demonstrated here

Any live researcher loop, any parallel-hypothesis yield, any audit independence score, Combined AGSi, or a METR eval.
