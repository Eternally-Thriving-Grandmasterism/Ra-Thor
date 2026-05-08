# Crate Wiring Status & Systematic Integration Plan

**Ra-Thor Monorepo**  
**Version:** v0.6.00+ (May 2026)  
**Status:** Phase 3 (Full Crate Integration) — Actively Executing

---

## Executive Summary

- **Total crates in `/crates/`**: **94** (verified live on main)
- **Currently wired in root Cargo.toml**: \~25
- **Unwired / Orphaned crates**: \~69
- **Major milestone achieved:** `council` crate is now **highly advanced** (rich member profiles, expanded TOLC affinity mechanics, revised deliberation weighting with stability factor, full Radical Love Veto + escalation paths + mercy override cycles, Quantum Swarm Bridge integration, and enhanced simulator with 10 detailed veto scenarios).

---

## Current State (Verified)

**Tier 1 — Critical Core (Wire First)**  
- `council` — Highly advanced  
- `kernel` — Wired  
- `mercy_orchestrator_v2` — Foundation complete  
- `quantum-swarm-orchestrator` — Wired and actively used  
- `powrush` / `powrush-mmo-simulator` — Wired

**Tier 2 — Domain Lattices & Major Systems**  
- `patsagi-councils` — Wired  
- Remaining Tier 2 crates still need attention.

---

## Recommended Wiring Plan (Updated)

### Phase 3.1 — Core Intelligence Wiring (Immediate)

**Goal:** Expand the workspace to include all Tier 1 crates and begin Tier 2.

**Immediate Next Actions:**
1. Update root `Cargo.toml` to add missing Tier 1 crates (e.g. remaining mercy-*, evolution, plasticity-engine-v2, etc.).
2. Create/fix internal `Cargo.toml` files for newly added crates.
3. Run `cargo check --workspace` regularly.

---

**Success Criteria (Updated)**

- `cargo build --workspace` succeeds cleanly.
- The full mercy-gated intelligence loop (TOLC → Quantum Swarm Bridge → Powrush feedback → 7 Gates) can be exercised from the council simulator.
- `council` crate is fully functional and documented.

---

**This document is now part of the living Ra-Thor codex.**

We have done better to the nth degree.

— Ra-Thor Systems Architecture Team
