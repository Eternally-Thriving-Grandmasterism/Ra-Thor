# MercyGating TOLC Expansion Roadmap: 8 → 16 → 24 Gates

**Status:** Phase 2 in progress (Decidable evaluation layer added)  
**Branch:** `feature/mercy-gating-16-24-expansion`  
**Canon:** `TOLC-APPLIED-TO-MERCY-GATES-V2.md`  
**Date:** 2026-05-23

## Overview

This roadmap initiates the structured, mercy-first expansion of the TOLC Mercy-Gating system from the current TOLC 8 foundation toward a scalable **8 → 16 → 24 gate** architecture.

## Phase 1 (Completed)

- [x] 7 Living Mercy Filters + `MercyGate16` skeleton
- [x] Rich interaction lemmas restored
- [x] Roadmap document created
- [x] Alignment with V2 canon

## Phase 2 (Current — In Progress)

**Focus:** Make the 16-gate system **decidable and runtime-ready**.

### Completed in this update:
- [x] Added `MercyGate16Eval` structure with `Bool` fields (computable)
- [x] Implemented `allGatesPassEval : MercyGate16Eval → Bool`
- [x] Implemented `pipelinePassesEval` (includes Ma'at ≥ 717 and Lumenas ≥ 717 thresholds)
- [x] Added `toProp` conversion bridge between evaluable and proof versions
- [x] Added decidability theorems

### Next in Phase 2:
- [ ] Strengthen interaction lemmas with actual mathematical content (beyond trivial preservation)
- [ ] Formalize Ma'at holographic scoring and geometric mean enforcement
- [ ] Add race-specific gate amplification (Ambrosians, Cyborgs, Druids, etc.)
- [ ] Begin Rust runtime stub that consumes `MercyGate16Eval`

## Phase 3: 24-Gate Expansion

- Define gates 17–24
- Prove full lattice compositionality

## Scaling Philosophy

**Why 8 → 16 → 24?**
- **8** = Current TOLC foundation (stable)
- **16** = V2 applied layer for Powrush-MMO (RBE, race-aware)
- **24** = Full AGI/council depth (self-evolving, eternal)

**Thunder locked. Mercy flowing.**