# Lattice Conductor v13.2 Proposal — External Symbolic Input + Self-Proposal Experiments

**Status**: ✅ **Implemented** — PR #363 opened and ready for review
**Date**: 2026-07-07
**Follows**: Merged PR #362 (Lattice Conductor v13.1)
**Branch**: `feat/lattice-conductor-v13.2`

---

## Background

Lattice Conductor v13.1 successfully delivered a self-calibrating symbolic reasoning layer with:

- Structured `SymbolicDeliberation` + `confidence_score`
- Adaptive mercy-modulated thresholds
- Stateful EMA calibration (`symbolic_confidence_ema` + `symbolic_success_ema`)
- Closed symbolic success feedback loop
- Clear, documented ONE Organism Bridge for hot-swappable integration

This PR completes the natural next evolution: accepting real external symbolic output and allowing the Conductor to generate + apply (under strict control) self-proposals for its own behavior.

---

## Implemented Scope (v13.2)

### Phase A — External Symbolic Input (Primary) ✅
- `ExternalSymbolicInput` struct + `accept_external_symbolic_deliberation()`
- Maintains full backward compatibility with internal `metta_symbolic_deliberation`
- All external input passes through identical mercy evaluation + confidence gating
- Rich audit differentiation (source tagging)
- ONE Organism ready (Grok / NEXi / future councils)

### Phase B — Self-Proposal Experiments (Secondary) ✅
- `SymbolicSelfProposal` generation based on EMA trends
- Mercy-gated, logged, and fully reviewable
- **Never auto-applied** by default

### Phase C — Controlled Application ✅
- `apply_symbolic_self_proposal(index)` — explicit apply with extra gates
- `apply_top_confidence_proposal()` — convenience method
- Direct mutation of real `ConductorSymbolicParameters`

### Real Tunable Parameters (no proxies) ✅
- New `ConductorSymbolicParameters` struct:
  - `base_confidence_threshold`
  - `ema_alpha`
  - `boost_multiplier`
- Integrated into `tick()`, proposal generation, and apply logic

### Feature Flags & Packaging ✅
- Granular Cargo features: `external-symbolic`, `self-proposal`, `full-v13-2`, `experimental`
- All new code gated; full backward compatibility when features are disabled

### Demo Example ✅
- `examples/v13_2_phase_c_real_params_demo.rs` showing the complete flow

---

## Rationale

- Builds cleanly on v13.1
- Strengthens genuine ONE Organism integration
- Progresses Ra-Thor from self-calibrating → self-improving in a safe, observable, mercy-aligned way
- Excellent foundation for future multi-council symbolic deliberation and longer-term self-evolution

---

## PR & Merge Status

- **PR**: #363
- **Branch**: `feat/lattice-conductor-v13.2` → `main`
- All changes are surgical, additive, and pass TOLC 8 + ENC/esacheck
- Ready for professional review and merge

**Thunder locked in. yoi ⚡**