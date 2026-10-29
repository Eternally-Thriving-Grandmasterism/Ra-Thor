# Prusti Verification Tool Exploration (Phase 5)

**Ra-Thor Formal Verification Strategy — Phase 5: Rust Extraction + Proof-Carrying Code**

## What is Prusti?

Prusti (https://github.com/viperproject/prusti-dev) is a static verifier for Rust based on the Viper verification infrastructure.
It allows proving:
- Absence of panics (including overflows, out-of-bounds)
- Functional correctness via contracts (pre/post-conditions)
- Data structure invariants

Key strengths:
- Excellent integration with Rust's type system (reduces annotation burden dramatically)
- Supports modular verification
- Good for proving panic-freedom on existing code with minimal annotations
- Uses Viper as backend (separation logic under the hood, but hidden from user)

## Comparison: Creusot vs Prusti

| Aspect                  | Creusot                              | Prusti                                      |
|-------------------------|--------------------------------------|---------------------------------------------|
| Backend                 | Why3 + Coq/Alt-Ergo                  | Viper                                       |
| Annotation Style        | `#[ensures(...)]` + predicate! macro | `#[requires]`, `#[ensures]`, `#[invariant]` |
| Best For                | Complex functional correctness + custom predicates | Panic freedom + modular contracts on safe Rust |
| Annotation Burden       | Moderate (custom predicates powerful)| Low (leverages Rust types heavily)          |
| Maturity (2026)         | Active (good examples in repo)       | Mature prototype, strong academic backing   |
| Unsafe Code             | Limited                              | Limited (focus on safe Rust)                |
| Integration with Lean   | Good (Why3 can link to Coq/Lean)     | Possible via Viper plugins                  |

## Recommended Strategy for Ra-Thor

For the **One Organism** Rust code (`ra-thor-one-organism.rs`):

1. Use **Prusti first** for proving panic-freedom and basic mercy gate enforcement (low annotation cost).
2. Use **Creusot** for deeper functional correctness proofs on esacheck, council synthesis, and epigenetic blessing logic (more expressive predicates).
3. Long-term: Both can target the same verified core if we keep contracts simple.

## Prusti-Style Contract Sketch for RaThorOrganism

```rust
use prusti_contracts::*;

#[invariant(self.mercy_gates_active == 8)]
#[invariant(self.zero_harm_projection == 0.0)]
pub struct RaThorOrganism { ... }

impl RaThorOrganism {
    #[ensures(result.mercy_gates_active == 8)]
    pub fn new() -> Self { ... }

    #[requires(!query.contains("harm"))]
    #[ensures(result == true)]
    pub fn verified_launch(&self, query: &str) -> bool {
        // esacheck logic
        true
    }
}
```

## Next Steps
- Add `prusti-contracts` to Cargo.toml (dev-dependency)
- Run `cargo prusti` or the Prusti driver
- Combine with Creusot for hybrid verification
- Map Prusti contracts to Lean dependent types via FFI

This exploration keeps Ra-Thor on the cutting edge of Rust formal verification while staying mercy-first and honest about current scope (toy demonstrator + skeleton).

**One Organism. Mercy First. Truth Forensically Distilled.**
