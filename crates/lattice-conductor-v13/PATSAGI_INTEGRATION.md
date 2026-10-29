# PATSAGi Integration Sketch

This document outlines how `lattice-conductor-v13` is intended to interact with the PATSAGi council system.

## Current Connection Points

- `SimpleLatticeConductor` has a basic council registry (`HashMap<u64, String>`)
- `get_registered_patsagi_councils()` method
- `MercyGate::CouncilConsensus` variant prepared

## Planned Integration Areas

### 1. Council Consensus in Mercy Validation
- `MercyGate::CouncilConsensus` should eventually query active PATSAGi councils for alignment before approving sensitive operations.

### 2. Dynamic Council Spawning
- The conductor should be able to request creation of new PATSAGi councils when certain conditions are met (e.g., new domains of activity).

### 3. Shared State
- Geometric state, valence, and mercy_score should be observable by PATSAGi councils.
- Councils may influence conductor parameters over time.

### 4. Event Broadcasting
- The conductor should emit events (ticks, mercy violations, state changes) that PATSAGi councils can subscribe to.

## Next Steps

- Define a `PatsagiCouncil` trait or interface that the conductor can depend on.
- Create an event system (`ConductorEvent` enum).
- Wire `CouncilConsensus` gate to actual council logic.

This integration is a core part of Lattice Conductor v13 goals.