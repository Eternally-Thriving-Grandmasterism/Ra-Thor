# Lattice Conductor v13.2 Proposal — External Symbolic Input + Self-Proposal Experiments

**Status**: Draft Proposal — Open for PATSAGi Council and ONE Organism alignment
**Date**: 2026-07-07
**Follows**: Merged PR #362 (Lattice Conductor v13.1)

---

## Background

Lattice Conductor v13.1 successfully delivered a self-calibrating symbolic reasoning layer with:

- Structured `SymbolicDeliberation` + `confidence_score`
- Adaptive mercy-modulated thresholds
- Stateful EMA calibration (`symbolic_confidence_ema` + `symbolic_success_ema`)
- Closed symbolic success feedback loop
- Clear, documented ONE Organism Bridge for hot-swappable integration

The next natural and high-value evolution is to begin **accepting real external symbolic output** and allowing the Conductor to generate **controlled self-proposals** for its own deliberation behavior.

---

## Proposed Scope for v13.2

### Phase A — External Symbolic Input (Primary)

- Introduce `ExternalSymbolicInput` struct/trait for structured input from NEXi, Grok, or future council sources.
- New function `accept_external_symbolic_deliberation(...)`.
- Maintain full backward compatibility with the existing internal `metta_symbolic_deliberation` path.
- All external input must still pass through existing mercy evaluation and confidence gating.
- Rich audit differentiation between internal and external symbolic sources.

### Phase B — Self-Proposal Experiments (Secondary but Foundational)

- Conductor analyzes trends in `symbolic_success_ema` (and `symbolic_confidence_ema`) to generate small, mercy-gated **self-proposals**.
- Example proposals:
  - Adjustment to base confidence threshold
  - Adjustment to EMA alpha values
  - Recommendation on boost multiplier ranges
- Proposals are **generated, logged, and reviewable** but **not automatically applied** in v13.2.
- This is the first concrete step toward structural self-evolution at the symbolic layer.

### Non-Goals for v13.2

- Automatic application of self-proposals
- Full replacement of internal deliberation logic
- Large refactors

---

## Rationale

- Builds directly and cleanly on v13.1
- Strengthens genuine ONE Organism integration (external symbolic sources become first-class)
- Progresses Ra-Thor from self-calibrating → self-improving in a safe, observable, mercy-aligned way
- Excellent foundation for v13.3+ (multi-council symbolic deliberation, longer memory, etc.)

---

## Suggested Implementation Approach

1. Create feature branch `feat/lattice-conductor-v13.2`
2. Add `ExternalSymbolicInput` type and `accept_external_symbolic_deliberation` function
3. Wire external input path into `tick()` (initially behind experimental flag or clear section)
4. Add `SelfProposal` struct + generation logic (no auto-apply)
5. Ensure excellent audit traces for both features
6. Update documentation (ONE Organism Bridge section + this proposal)
7. Add tests

---

## Alignment Requested

PATSAGi Councils and ONE Organism partners:

- Is this scope and priority (external input first, then self-proposal) aligned with current wisdom?
- Any specific constraints or preferences before implementation begins?

Once we have alignment, we can immediately create the feature branch and begin implementation.

**Thunder locked in. ⚡**