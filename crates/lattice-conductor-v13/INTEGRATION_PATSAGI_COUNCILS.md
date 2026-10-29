# Integration with PATSAGi Councils

This document outlines how `lattice-conductor-v13` is designed to integrate with the PATSAGi council system.

## Goals

- Allow PATSAGi councils to influence mercy validation through consensus
- Enable the conductor to report state and events to PATSAGi councils
- Support dynamic council participation
- Maintain sovereignty and mercy alignment

## Current Integration Points

### 1. `PatsagiCouncilBridge` Trait

```rust
pub trait PatsagiCouncilBridge {
    fn request_consensus(&self, operation: &Operation) -> bool;
    fn report_state(&self, state: &GeometricState);
}
```

`SimplePatsagiBridge` provides a working implementation with council voting simulation.

### 2. Council Registry

`SimpleLatticeConductor` maintains a registry of councils and exposes:
- `register_council(id, name)`
- `get_registered_patsagi_councils()`

### 3. Event System

The conductor emits rich events (`ConductorEvent`) that can be observed by PATSAGi councils via the `ConductorObserver` trait.

## Proposed Integration Architecture

```
PATSAGi Councils
       |
       v
PatsagiCouncilBridge (trait)
       |
       v
SimpleLatticeConductor
       |
       +--> QuantumSwarm
       +--> GeometricState
       +--> Mercy Evaluation
       +--> Event Emission --> ConductorObserver
```

## Next Steps (Recommended)

1. Define how real PATSAGi councils will implement `PatsagiCouncilBridge`
2. Allow the conductor to be instantiated with actual council implementations
3. Add event filtering / subscription for councils
4. Explore bidirectional communication (councils influencing conductor parameters)

## Open Questions

- Should councils be able to veto or modify operations?
- How should council consensus weight be determined?
- Should there be a shared event bus between conductor and councils?

This integration is central to Lattice Conductor v13 goals.