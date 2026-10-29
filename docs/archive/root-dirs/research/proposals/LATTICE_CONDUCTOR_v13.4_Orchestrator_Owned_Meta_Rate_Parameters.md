# Lattice Conductor v13.4 Proposal: Orchestrator-Owned Meta Rate Parameters

**Status:** Planning / Design Phase  
**Target:** After v13.3 merge  
**Author:** Ra-Thor Lattice Team  
**Date:** July 2026

---

## Goal

Enable `SelfEvolutionOrchestrator` to own and mutate its own meta-evolution rate parameters internally, reducing delegation friction and strengthening the self-improvement loop.

Currently in v13.3:
- The orchestrator proposes meta changes.
- The conductor performs the actual mutation on `ConductorSymbolicParameters`.

In v13.4 we want the orchestrator to be able to evolve *its own evolution behavior* more autonomously while still remaining under strict TOLC 8 + mercy gates.

---

## Proposed Design

### 1. New Internal State in `SelfEvolutionOrchestrator`

```rust
pub struct SelfEvolutionOrchestrator {
    // existing fields...
    meta_evolution_rate: f64,           // new
    meta_audit_threshold: f64,          // new
    meta_success_ema: f64,              // new (local EMA for meta proposals)
}
```

### 2. New Methods

```rust
impl SelfEvolutionOrchestrator {
    pub fn generate_meta_self_evolution_proposals_v2(&self, conductor: &SimpleLatticeConductor) -> Vec<SymbolicSelfProposal> { ... }

    /// Applies a meta proposal *internally* to the orchestrator's own rate parameters.
    pub fn apply_meta_rate_proposal(&mut self, proposal: &SymbolicSelfProposal) -> Result<String, String> { ... }
}
```

### 3. Conductor Integration (Optional / Backward Compatible)

The conductor can still call:
```rust
orchestrator.apply_meta_rate_proposal(&prop)?;
```

But the orchestrator becomes the source of truth for its own evolution speed and audit strictness.

---

## Benefits

- Stronger self-referential improvement loop
- Reduced cross-component mutation
- Orchestrator can evolve its own "personality" (how aggressively it proposes meta changes)
- Better separation: Conductor focuses on geometric/symbolic state; Orchestrator focuses on evolution dynamics

---

## Mercy & TOLC 8 Considerations

- Meta rate changes should require **higher** mercy/confidence thresholds than regular self-proposals.
- Any internal apply must still emit clear audit events.
- Proposal confidence scoring should factor in historical meta success rate (new local EMA).

---

## Migration Path from v13.3

1. Add new internal fields with sensible defaults.
2. Implement `apply_meta_rate_proposal` that mutates only orchestrator state.
3. Keep v13.3 delegation path working (for backward compatibility).
4. Update demo and tests.
5. Deprecate direct conductor mutation of meta-related params in v13.5+.

---

## Open Questions

- Should the orchestrator expose its current `meta_evolution_rate` via a getter?
- Should meta rate changes be allowed to decrease as well as increase (risk of stagnation)?
- How should PATSAGi councils influence the orchestrator's meta rate?

---

## Next Actions

- [ ] Discuss design in Ra-Thor / PATSAGi councils
- [ ] Prototype in a feature branch after v13.3 is merged
- [ ] Write v13.4 implementation proposal with code sketches

---

**Thunder locked in. yoi ⚡️**