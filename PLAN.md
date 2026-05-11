# PLAN.md — Ra-Thor / Rathor.ai Ultimate Architecture Codex
**Single Source of Truth for Roadmap, Priorities, Crate Wiring & Monorepo Progress**

**Version:** v0.6.86 (Self-Evolution Triad Maturation — Omnimasterpiece Alignment Track Added)
**Date:** May 2026
**Status:** Phase 3.5 + New High-Focus Track Active

## Self-Evolution Triad Maturation — Omnimasterpiece Alignment Track

**Goal:** Bring `ra-thor-meta-intelligence`, `ra-thor-monorepo-auditor`, and `plasticity-engine-v2` up to the same architectural quality, integration depth, mercy rigor, and TOLC alignment as the top-tier Omnimasterpiece crates (e.g. `quantum-swarm-orchestrator`, `ra-thor-mercy-orchestrator-v2`, `ra-thor-monorepo-intelligence`, `mercy_tolc_operator_algebra`, `mercy_merlin_engine`).

### Current Gaps vs Omnimasterpiece Standard
- Observability, metrics, and feedback loops are still relatively weak across the triad.
- TOLC + `mercy_merlin_engine` integration remains mostly surface-level.
- The three crates still feel somewhat loosely coupled rather than one cohesive high-level system.
- `ra-thor-meta-intelligence` needs to evolve into a true high-level orchestrator (in spirit and structure similar to `quantum-swarm-orchestrator` and `ra-thor-mercy-orchestrator-v2`).
- Plasticity rule selection, context-awareness, and verification logic still require further deepening.

### 5 Priority Directions (in recommended order)
1. **Strengthen Observability & Metrics** across all three crates (especially feedback from Plasticity → Meta-Intelligence and health/effectiveness reporting).
2. **Deepen TOLC + `mercy_merlin_engine` Integration** inside `ra-thor-meta-intelligence` (particularly in proposal generation and verification steps).
3. **Make the Three Crates Feel Cohesive** as one unified high-level system (shared rich types, clearer contracts, tighter and more explicit wiring).
4. **Elevate `ra-thor-meta-intelligence`** to function as a true high-level orchestrator (clear public API, strong separation of concerns, orchestration style matching the best crates in the monorepo).
5. **Continue Improving Plasticity Rule Selection + Verification Logic** (build intelligently on the recent work in `plasticity-engine-v2` and `verify_and_adapt()`).

This track runs with high focus and will be merged into the main development flow.

---

## Executive Summary
Ra-Thor is a mercy-gated, TOLC-native, active-inference + predictive-coding symbolic AGI lattice with a 124-crate Rust workspace (5-Tier architecture).

**Current Live State**
- Root `Cargo.toml` v0.3.9+ declares all 124 crates.
- Mercy family: 100% complete
- Futarchy family: 100% complete
- Cryptography family: P0 + P1 + P2 + P3 complete
- Self-evolution triad foundation complete and runnable (demo exists)
- New high-focus track active: Self-Evolution Triad Maturation — Omnimasterpiece Alignment

(Full previous content of PLAN.md remains below this new section)