# SpacetimeDB Reducer Bridge

Mercy-gated adapter for SpacetimeDB-style reducers (WASM) inside the Ra-Thor Lattice Conductor v14.

**Purpose**: Execute transactional game logic (reducers) with full TOLC 8 Mercy Gates, MIAL/MWPO scoring, and Thunder Lattice Governance.

**Key Features**
- Atomic reducer execution (SpacetimeDB semantics)
- Pre-execution mercy check (valence ≥ 0.999999)
- Post-execution valence & norm preservation verification
- Automatic integration into Cosmic Self-Evolution Loop
- PATSAGi Council parallel review

**Usage**
```rust
let bridge = SpacetimeReducerBridge::new();
let result = bridge.execute_reducer("powrush_terrain_edit", payload).await?;
