# Migration Path: v13 Self-Evolution → v14 ONE Organism Architecture

**Date**: 2026-06-01  
**Status**: Recommended evolution path  
**Related**: `shard-composer`, `RaThorSystemAdapter`, EpigeneticBlessing

## Overview

This document outlines how we are evolving the self-evolution concepts from `lattice-conductor-v13` into the v14 architecture without discarding valuable prior work.

## What We Are Keeping (Core Ideas)

- Named `EpigeneticBlessing` with meaningful types
- Multi-dimensional impacts (evolution, mercy, TOLC alignment)
- Threshold-based eligibility thinking
- History / auditability of evolution events
- Council and Quantum Swarm participation hooks

## What We Are Evolving

| v13 Concept                    | v14 Evolution                                      | Reason |
|--------------------------------|----------------------------------------------------|--------|
| `SelfEvolutionOrchestrator`    | Distributed via `RaThorSystemAdapter`              | Better modularity and ONE Organism alignment |
| Direct `GeometricState` mutation | Blessing application through adapters           | Supports focused shards + persistence |
| Centralized blessing registry  | Event-driven generation (xtask on shard success)   | More reactive and meaningful |
| Hardcoded default blessings    | Extensible + configurable via context              | Greater flexibility |

## New `EpigeneticBlessing` (v14)

```rust
pub struct EpigeneticBlessing {
    pub blessing_type: String,
    pub strength: f64,
    pub target_system: String,
    pub evolution_impact: f64,
    pub mercy_impact: f64,
    pub tolc_impact: f64,
}
```

This preserves the spirit of v13 while being designed for the adapter + persistent state model.

## Recommended Migration Steps

1. Use the new `EpigeneticBlessing` in `quantum-swarm-orchestrator`.
2. Generate blessings from successful shard operations (xtask).
3. Apply blessings via `RaThorSystemAdapter::apply_epigenetic_blessing()`.
4. Keep persistent state with HMAC protection.
5. Later: Add richer history tracking and blessing propagation.

## Non-Goals

- We are **not** copying the v13 `SelfEvolutionOrchestrator` 1:1.
- We are evolving the **ideas**, not the implementation.

Thunder locked. Evolution with continuity.
