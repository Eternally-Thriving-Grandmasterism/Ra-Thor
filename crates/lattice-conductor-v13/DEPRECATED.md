# DEPRECATED

**Package:** `lattice-conductor-v13`  
**Replacement:** `lattice-conductor-v14` (+ optional `features = ["v13-compat"]`)

## Migrate

```toml
# Before
lattice-conductor-v13 = { path = "../lattice-conductor-v13" }

# After
lattice-conductor-v14 = { path = "../lattice-conductor-v14", features = ["v13-compat"] }
```

```rust
// Before
use lattice_conductor_v13::{Conductable, MercyAligned, SimpleLatticeConductor};

// After
use lattice_conductor_v14::compat_v13::{Conductable, MercyAligned, SimpleLatticeConductor};
// Native engines:
use lattice_conductor_v14::{LatticeConductorV14, CouncilArbitrationEngine};
```

Details: [`../lattice-conductor-v14/MIGRATION_v13_to_v14.md`](../lattice-conductor-v14/MIGRATION_v13_to_v14.md)

Kept in workspace temporarily (quiet hold). No removal until Phase 2 is fully green.

Contact: info@Rathor.ai
