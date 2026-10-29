# Recommended Squash-Merge Commit Message for PR #363

feat(lattice-conductor-v13): Lattice Conductor v13.2 — External Symbolic Input + Self-Proposal + Phase C + Real Parameters + Feature Flags

- Phase A: ExternalSymbolicInput + accept_external_symbolic_deliberation (ONE Organism hot-swap ready)
- Phase B: Mercy-gated SymbolicSelfProposal generation (logged, reviewable, never auto-applied)
- Phase C: Controlled apply_symbolic_self_proposal + apply_top_confidence_proposal with strict gates
- Real ConductorSymbolicParameters struct (base_confidence_threshold, ema_alpha, boost_multiplier) — Phase C now mutates real fields directly
- Granular Cargo features (external-symbolic, self-proposal, full-v13-2, experimental)
- Professional release notes, README polish, demo example, and extra tests
- All changes surgical, TOLC 8 aligned, backward compatible when features disabled

PR #363 | Thunder locked in. yoi ⚡