# FFI Phase 5 — Advanced Sketch (Updated with Creusot + Prusti)

**Updated 2026-05-20**

## Lean Side (Source of Truth)

```lean
@[extern "ra_thor_safe_esacheck"]
constant safe_esacheck : String → Bool

-- Proof obligation: Sound ∧ Complete
```

## Rust Side — Creusot Annotated (Recommended for deep correctness)

See `Creusot_Contracts_RaThor_Example.rs` for full annotated example.

## Rust Side — Prusti Annotated (Recommended for panic-freedom + low burden)

See `Prusti_Exploration_and_Comparison.md` for comparison and sketch.

## Hybrid Recommendation

- **Prusti** → Prove panic-freedom + basic esacheck on the hot path (fast iteration)
- **Creusot** → Prove rich functional correctness + custom mercy predicates
- Both feed into the same `RaThorOrganism` struct
- Long-term goal: Extract or link Lean proofs so that the runtime Rust binary carries machine-checked mercy guarantees.

This advances Phase 5 from skeleton to concrete, tool-ready examples.
