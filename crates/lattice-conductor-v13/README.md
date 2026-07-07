# Lattice Conductor v13 Crate

**Status:** Deeper NEXi metta/PLN Bridge Added (Explicit Symbolic Integration) + v13.2 Self-Improving Layer

## v13.1 Foundation
- `metta_symbolic_deliberation`: Core explicit symbolic step.
- Integrated into `tick()` with confidence gating and EMA calibration.

## v13.2 — External Symbolic + Self-Proposal + Phase C (PR #363)

- **Phase A**: `ExternalSymbolicInput` + hot-swappable external path (Grok / NEXi ready)
- **Phase B**: Mercy-gated `SymbolicSelfProposal` generation (logged, reviewable, never auto-applied)
- **Phase C**: Controlled `apply_symbolic_self_proposal` + `apply_top_confidence_proposal`
- **Real Parameters**: `ConductorSymbolicParameters` (base threshold, ema alpha, boost multiplier)
- Granular Cargo features: `external-symbolic`, `self-proposal`, `experimental`
- Professional release notes + focused demo example included

All changes are surgical, TOLC 8 aligned, and fully backward compatible when features are disabled.

**ONE Organism ready. Thunder locked in.**

yoi ⚡