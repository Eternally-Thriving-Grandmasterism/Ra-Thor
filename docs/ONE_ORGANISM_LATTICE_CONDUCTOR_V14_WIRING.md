# ONE Organism ↔ Lattice Conductor v14 — Wiring Guide

**Status:** TRUE PATH DEPENDENCY LIVE  
**Version:** Organism Core **14.9.0** / lattice-conductor-v14 **14.8.2**  
**Date:** 2026-07-19

## 1. Workspace Crate (preferred)

```toml
# crates/ra-thor-one-organism/Cargo.toml
[dependencies]
lattice-conductor-v14 = { path = "../lattice-conductor-v14", version = "14.8.2" }
```

```rust
use ra_thor_one_organism::{
    launch_one_organism_core,
    OneOrganismCore,
    CouncilArbitrationEngine,   // re-exported from lattice-conductor-v14
    RuntimeSelfHealingEngine,   // re-exported from lattice-conductor-v14
    Diagnosis, HealingAction,
};

let mut core = launch_one_organism_core();
assert!(core.is_cosmic_loop_ready());
```

## 2. Shared Cosmic Loop Flag Contract

```
CouncilArbitrationEngine::cosmic_loop_flag()  → Arc<AtomicBool>
        │
        ├─ RuntimeSelfHealingEngine.cosmic_loop_ready   (same Arc)
        ├─ LatticeConductorV14.cosmic_loop_ready         (same Arc)
        └─ OneOrganismCore.cosmic_loop_ready             (same Arc)
```

No local compatibility reimplementation. Types come from `lattice-conductor-v14`.

## 3. Historical Root File

`ra-thor-one-organism.rs` (repo root) remains as the **extended surface**
(GPU pipeline, GitHub connector, Quantum Swarm, full RoleOrchestrator).
It still carries a local compatibility block for standalone reference;
the **production path** is `crates/ra-thor-one-organism`.

## 4. Architectural Guarantee

```
launch_one_organism_core()
        │
        ▼
OneOrganismCore (v14.9.0)
  ├─ arbitration_engine: CouncilArbitrationEngine   (from lattice-v14)
  ├─ self_healing_engine: RuntimeSelfHealingEngine  (from lattice-v14)
  ├─ lattice: LatticeConductorV14
  ├─ cosmic_loop_ready: Arc<AtomicBool>             (shared)
  └─ active_role / shared_valence
```

**Thunder locked in.** yoi ⚡
