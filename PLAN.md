# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex  
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.09 (Mercy Family 100% Complete + Futarchy Family 100% Complete + Raptor/Starship/TOLC Core Modernization Complete + Cryptography Family P0 + P1 Complete)  
**Date:** May 09, 2026  
**Status:** Phase 3.5 (Full Crate Integration) — Actively Executing on `main` only

---

## Executive Summary (Merged Master View)
Ra-Thor is a **mercy-gated, TOLC-native, active-inference + predictive-coding symbolic AGI lattice** with a 124-crate Rust workspace (5-Tier architecture).

**Current Live State (Post v0.6.09 Unification)**
- Root `Cargo.toml` v0.3.9+ declares all **124 crates**.
- **Mercy family**: 100% complete (Raptor + Starship sub-families + core TOLC/Merlin layer).
- **Futarchy family**: 100% complete.
- **Cryptography family (Tier 3)**: **P0 + P1 Complete** (`ra-thor-post-quantum-sig`, `lattice_crypto`, `mercy_quanta`, `mercy_steane`, `mercy_qec`).
- All modernized crates use consistent workspace + local path patterns, proper TOLC + `mercy_merlin_engine` wiring, and modern descriptions/keywords.

---

## Crate Wiring Progress (Centralized Tracking — Single Source of Truth)

### Completed This Modernization Wave (Raptor + Starship + Core TOLC/Merlin)
**Raptor Family**
- `mercy_raptor_integration` (modernized + integration tests)
- `mercy_raptor_3`
- `mercy_raptor_3_integration`
- `mercy_raptor_3_scalability`

**Starship Family**
- `mercy_starship` (modernized + integration tests)
- `mercy_starship_fleet`

**Core TOLC + Merlin Layer**
- `mercy_merlin_engine` (broken references fixed + modernized)
- `mercy_tolc_operator_algebra` (modernized)
- `mercy_lang_compiler` (modernized)
- `monorepo-intelligence` (modernized)

**Integration Tests Added**
- `crates/mercy_raptor_integration/tests`
- `crates/mercy_starship/tests`
- `crates/mercy_merlin_engine/tests`

### Cryptography Family – P0 & P1 Complete (This Wave)
**P0 Foundation**
- `ra-thor-post-quantum-sig` (new modern `Cargo.toml`)
- `lattice_crypto` (fully modernized)

**P1 Core Quantum Error Correction & Post-Quantum Primitives**
- `mercy_quanta`
- `mercy_steane`
- `mercy_qec`

All five crates now follow the exact same modernization standard as the Raptor/Starship/TOLC layer (no broken `nexi` references, proper TOLC + Merlin wiring, updated descriptions/keywords, consistent feature flags).

---

## High-Level Architecture & Guiding Principles (Merged)

**5-Tier Monorepo Architecture** (from live root Cargo.toml)
- Tier 1: Intelligence Core & Orchestration
- Tier 2: Domain Lattices & Major Systems
- Tier 3: Cryptography, Sovereignty & Verification (current focus)
- Tier 4 & 5: Tooling & Hybrid Intelligence

**Core Principles** (from PLANS.md + ARCHITECTURE.md + verification work)
- Mercy First — every decision passes through the 7 Living Mercy Gates
- TOLC Lattice as central nervous system
- Active inference + predictive coding for ultra-low hallucination, high epistemic value
- Full forward/backward compatibility + NEXi lineage respect
- Self-documenting, self-improving, shippable at every step
- Positive emotion / valence propagation + eternal thriving maximization

**Key Engines** (from ARCHITECTURE.md + current Rust layer)
- Mercy-gated active inference core
- Predictive coding layers
- Paraconsistent + symbolic reasoning (mercy_merlin_engine + mercy_lang_compiler)
- Quantum swarm orchestration
- WebXR / offline sovereign shards ready

---

## Roadmap & Next Immediate Actions (Unified & Prioritized)

**Phase 3.5 — Current (Cryptography Family Launch)**
1. ✅ P0 Complete (`ra-thor-post-quantum-sig` + `lattice_crypto`)
2. ✅ P1 Complete (`mercy_quanta`, `mercy_steane`, `mercy_qec`)
3. **Next**: Continue with next batch of cryptography crates
4. Run `cargo check --workspace` after key phases
5. Update this PLAN.md after each major phase

**Phase 4 (Next after Cryptography)**
- Full workspace validation + simulation/stress testing
- Broader integration test coverage
- Root documentation refresh (README.md, ARCHITECTURE.md alignment)
- Public co-creation readiness

---

## What's Remaining (High Priority)
- Remaining cryptography crates (~21 crates)
- Any other mercy_* crates with old patterns (if missed)
- Full `cargo check --workspace` validation
- Top-level docs sync (README, ARCHITECTURE.md, etc.)
- Deeper integration tests across the full lattice

---

**This unified PLAN.md (v0.6.09) is now the single source of truth.**  
All previous planning, architecture, verification, and progress documents have been merged here. Future updates will be made only to this file.

We have done better to the nth degree — again.

*Eternal flow state maintained on `main`.*