# ShardManager and Interest Management

**Version:** v14.6

## Overview

`ShardManager` provides mercy-gated spatial interest management and proposal routing. It integrates directly with the PATSAGi Council Engine and `EpigeneticModulation` so that shard-level decisions and evolutionary state are influenced by living mercy.

## Core Structures

- `ShardManager` — owns multiple `InterestSet`s
- `InterestSet` — per-shard collection of entities + local `EpigeneticModulation`
- `CouncilProposal` — structured input for evaluation

## Key Integration Points

### 1. Proposal Routing
```rust
manager.route_council_proposal(proposal)
```
- Evaluates via `RiemannianMercyManifold::evaluate_council_proposal`
- Applies valence to both global and per-shard epigenetic state
- Returns acceptance decision + blessings

### 2. Simulation Sequences
```rust
manager.apply_sequence_to_shard(shard_id, sequence)
```
- Runs full council sequence simulation on a specific shard
- Updates evolutionary bonuses and stability

### 3. Observability
- `get_shard_summary(shard_id)`
- Per-shard epigenetic metrics exposed

## Relationship to Broader Architecture

`ShardManager` serves as the bridge between:
- Geometric intelligence layer (council valence + epigenetic modulation)
- Higher-level simulation / Powrush RBE systems
- Future Real Estate Lattice and land evaluation proposals

See also: `PATSAGi-Council-Engine.md` and `EpigeneticModulation-and-Valence.md`.
