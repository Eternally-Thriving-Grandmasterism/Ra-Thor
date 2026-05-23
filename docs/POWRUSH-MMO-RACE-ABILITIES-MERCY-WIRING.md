# Powrush-MMO Race Abilities ↔ Mercy Gates Wiring

**Status: Active Bridge Layer**

This document defines how Powrush-MMO race abilities directly interface with the MercyGating runtime and Lean formal model.

## Core Races & Abilities

### Druid
- **Ecosystem Mastery**: Amplifies `ecosystem`, `sustainability`, `harmony` gates by up to 1.28×
- Triggers extra Ma'at geometric mean boost
- Effect: Terrain and ally healing fields scale with current Mercy score

### Cyborg
- **Reversibility Shield**: Feeds `veracity` gate + provides temporary valence floor protection
- Reversibility charges protect against harm vector spikes

### Ambrosian
- **Joy Cascade**: Amplifies `joyfirst` and laughter-linked gates
- Cascades positive emotion flow to nearby beings

### Starborn
- **Infinite Potential**: Highest multipliers on `infinitepotential` and `eternalflow`
- Revelation charges unlock higher gate thresholds dynamically

## Runtime Wiring
See `crates/mercy_gating_runtime/src/powrush_mmo_race_abilities.rs` (to be expanded)

**ONE Organism enforcement active.**
