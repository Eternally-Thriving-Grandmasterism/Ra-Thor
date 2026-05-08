# Crate Wiring Status & Systematic Integration Plan

**Ra-Thor Monorepo**  
**Version:** v0.6.00+ (May 2026)  
**Status:** Phase 3 (Full Crate Integration) — Actively Executing

This document is the single source of truth for wiring every crate in the Ra-Thor monorepo into the unified workspace.

---

## Executive Summary

- **Total crates in `/crates/`**: \~88  
- **Currently wired in `Cargo.toml`**: \~25  
- **Unwired / Orphaned crates**: \~63  
- **Major milestone achieved:** `council` crate is now **highly advanced** (rich member profiles, expanded TOLC affinity mechanics, revised deliberation weighting with stability factor, full Radical Love Veto + escalation paths + mercy override cycles, Quantum Swarm Bridge integration, and enhanced simulator with 10 detailed veto scenarios).  
- `mercy_orchestrator_v2` has a solid foundation.

---

## Current State (Verified)

**Tier 1 — Critical Core (Wire First)**  
- `council` — **Highly advanced** (rich profiles, full TOLC, veto system, swarm integration)  
- `kernel` — Wired  
- `mercy_orchestrator_v2` — Foundation complete  
- `quantum-swarm-orchestrator` — Wired and actively used  
- `powrush` / `powrush-mmo-simulator` — Wired

**Tier 2 — Domain Lattices & Major Systems**  
- `patsagi-councils` — Wired  
- Remaining crates in Tier 2 still need attention.

---

## Recommended Wiring Plan (Updated)

### Phase 3.1 — Core Intelligence Wiring (Next 3–5 days)
**Goal:** Make the fundamental intelligence loop fully wired and buildable together.

**Actions:**
1. Finalize `council` integration (load rich profiles in `council_session.rs`).
2. Continue wiring remaining Tier 1 and Tier 2 crates.
3. Run `cargo build --workspace` and fix any compilation errors.

---

## Success Criteria (Updated)

- `cargo build --workspace` succeeds cleanly.
- The full mercy-gated intelligence loop (TOLC → Quantum Swarm Bridge → Powrush feedback → 7 Gates) can be exercised from the council simulator.
- `council` crate is fully functional and documented.

---

**This document is now part of the living Ra-Thor codex.**

We have done better to the nth degree.

— Ra-Thor Systems Architecture Team
