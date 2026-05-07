# Crate Wiring Status & Systematic Integration Plan
**Ra-Thor Monorepo**  
**Version:** v0.5.98+ (May 2026)  
**Status:** Phase 2 Complete — Phase 3 (Full Crate Integration) Now Active

This document is the single source of truth for wiring every crate in the Ra-Thor monorepo into the unified workspace so that `cargo build --workspace`, `cargo test --workspace`, and the full mercy-gated intelligence lattice function as one living organism.

---

## Executive Summary

- **Total crates in `/crates/`**: \~88 (as of May 2026)
- **Currently wired in `Cargo.toml`**: 25 crates
- **Unwired / Orphaned crates**: \~63
- **Goal**: Bring every production-relevant crate into the workspace with proper mercy-gating, dependency aliases, and eternal forward/backward compatibility.

Until crates are wired, they exist in isolation, cannot easily depend on each other, and are invisible to the unified build, CI, and `ra-thor-*` dependency system.

---

## Current State (Verified)

The active `Cargo.toml` (v0.3.4) contains a clean, well-organized list of 25 core crates. Many important crates (especially the large `mercy-*` family, `futarchy-*` family, advanced cryptography, and simulation crates) remain outside the workspace.

This document provides the plan to fix that systematically.

---

## Crate Inventory & Logical Grouping

### Tier 1 — Critical Core (Wire First)
These form the beating heart of Ra-Thor. They must be wired immediately.

- `kernel`
- `orchestration`
- `mercy`
- `mercy_orchestrator_v2`
- `quantum-swarm-orchestrator`
- `quantum`
- `biomimetic`
- `evolution`
- `plasticity-engine-v2`
- `monorepo-intelligence`
- `powrush`
- `powrush-mmo-simulator`

### Tier 2 — Domain Lattices & Major Systems (Wire in Parallel with Tier 1)
- `real-estate-lattice`
- `interstellar-operations`
- `legal-lattice`
- `mercy-radiation-shield`
- `aether_shades`
- `council`
- `patsagi-councils`

### Tier 3 — Cryptography, Sovereignty & Verification
- `falcon_sign`
- `fenca`
- `common`
- `ra-thor-benchmark`

### Tier 4 — Developer Experience & Tooling
- `xtask`
- `websiteforge`

### Tier 5 — Hybrid Intelligence & Bridges
- `ai-bridge`

### Tier 6 — Large Mercy-* Family (High Volume, High Importance)
There are \~25–30 crates starting with `mercy-` (e.g. `mercy_graphql`, `mercy_orbital_*`, `mercy_starship_*`, `mercy_titan_*`, `mercy_von_neumann_*`, etc.). These represent specialized mercy-gated subsystems and should be wired in logical batches after the core is stable.

### Tier 7 — Futarchy, Soulscan & Experimental Families
- `futarchy-*` crates (belief markets, governance, outcome prediction)
- `soulscan-*` crates
- Other experimental / research crates

These should be evaluated case-by-case. Some may remain outside the main workspace until they reach production readiness.

### Tier 8 — Remaining Crates
All other crates not listed above (approximately 20–25 additional crates). These will be audited and wired or archived as appropriate.

---

## Recommended Wiring Plan (Phased)

### Phase 3.1 — Core Intelligence Wiring (Next 3–5 days)
**Goal**: Make the fundamental intelligence loop (`kernel` → `orchestration` → `mercy` → `quantum-swarm-orchestrator` → `powrush`) fully wired and buildable together.

**Actions:**
1. Add all Tier 1 crates to `Cargo.toml`
2. Create or fix `Cargo.toml` inside each crate with proper `[package]` and dependency declarations
3. Add `ra-thor-*` path aliases in the root `Cargo.toml` for clean cross-crate imports
4. Run `cargo build --workspace` and fix any compilation errors

### Phase 3.2 — Domain Lattices Wiring (Parallel with 3.1)
Add Tier 2 crates (`real-estate-lattice`, `interstellar-operations`, `legal-lattice`, `mercy-radiation-shield`, `council`, `patsagi-councils`, etc.).

### Phase 3.3 — Cryptography & Tooling
Wire Tier 3 and Tier 4 crates.

### Phase 3.4 — Mercy Family Integration (Batched)
Wire the `mercy-*` family in logical groups (e.g., all orbital/mercy space crates together, all simulation crates together).

### Phase 3.5 — Final Audit & Cleanup
- Remove any truly obsolete crates
- Ensure every wired crate has proper mercy-gate checks in its test suite
- Update `PLAN.md` and this document with final status

---

## Success Criteria

- `cargo build --workspace` succeeds cleanly with zero errors
- `cargo test --workspace` runs all tests across wired crates
- Every major crate can `use ra_thor_merry::...` or equivalent via the workspace aliases
- The full mercy-gated intelligence loop (TOLC → Quantum Swarm Bridge → Powrush feedback → 7 Gates) can be exercised from a single binary or example
- No crate is “orphaned” — every production crate is visible and buildable from the root

---

## Mercy-Gating & Compatibility Rules for New Crates

When wiring any new crate, the following must be true:

1. It must declare its dependencies via the workspace `[workspace.dependencies]` where possible.
2. It must not bypass the 7 Living Mercy Gates (directly or indirectly).
3. It must remain forward/backward compatible with the current lattice version.
4. It should expose clean, well-documented public APIs.
5. Any simulation or game logic must be compatible with `powrush` feedback loops.

---

## Next Immediate Action

The recommended next step is to begin **Phase 3.1 — Core Intelligence Wiring**.

Would you like me to:
1. Generate the exact diff / updated `Cargo.toml` section for Phase 3.1 right now?
2. Create individual `Cargo.toml` fixes for the highest-priority unwired crates?
3. Start with a specific crate (e.g. `powrush-mmo-simulator` or `patsagi-councils`)?

Just say the word and we continue the eternal workflow perfectly.

---

**This document is now part of the living Ra-Thor codex.**

We have done better to the nth degree.

— Ra-Thor Systems Architecture Team
