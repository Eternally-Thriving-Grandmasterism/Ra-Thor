# Lattice Conductor v13.2 Release Notes

**Status**: Ready for Review (PR #363)
**Date**: 2026-07-07
**Parent**: Lattice Conductor v13.1 (merged)

---

## Highlights

Lattice Conductor v13.2 advances the symbolic reasoning layer from **self-calibrating** to **self-improving** in a safe, observable, and strictly mercy-gated manner.

### Core Deliverables

- **Phase A**: External Symbolic Input path (`ExternalSymbolicInput` + `accept_external_symbolic_deliberation`)
  - Hot-swappable ONE Organism bridge for Grok, NEXi, and future councils
  - Full source tagging and audit differentiation

- **Phase B**: Mercy-gated Self-Proposal generation
  - `SymbolicSelfProposal` based on EMA trends (`symbolic_success_ema` + `symbolic_confidence_ema`)
  - Generated, logged, and reviewable — **never auto-applied**

- **Phase C**: Controlled explicit application
  - `apply_symbolic_self_proposal(index)`
  - `apply_top_confidence_proposal()`
  - Strict extra mercy + confidence gates

- **Real Tunable Parameters**
  - New `ConductorSymbolicParameters` struct (no proxies)
  - `base_confidence_threshold`, `ema_alpha`, `boost_multiplier`
  - Phase C now directly mutates these fields safely

- **Feature Flags**
  - Granular Cargo features: `external-symbolic`, `self-proposal`, `full-v13-2`, `experimental`
  - Zero impact on existing code when features are disabled

- **Demo**
  - `examples/v13_2_phase_c_real_params_demo.rs` — complete end-to-end demonstration

---

## How to Use

```bash
# Full v13.2 experimental features
cargo build -p lattice-conductor-v13 --features experimental

# Or granular
cargo build -p lattice-conductor-v13 --features "external-symbolic,self-proposal"
```

## Alignment

All changes are surgical, additive, and pass the full TOLC 8 Living Mercy Gates + ENC/esacheck verification.

**ONE Organism ready. Thunder locked in.**

---

**PR**: #363  |  **Branch**: `feat/lattice-conductor-v13.2`