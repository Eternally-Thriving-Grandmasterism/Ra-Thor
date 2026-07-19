# ONE Organism ↔ Lattice Conductor v14 — Full Cargo Dependency Wiring

**Status:** Ready for activation  
**Version:** 14.91  
**Date:** 2026-07-19

## 1. Full Cargo Dependency Wiring

`lattice-conductor-v14` is already a workspace member.

To make the **real** crate the source of truth (instead of any thin local guardian), the package that owns / compiles `ra-thor-one-organism.rs` must declare:

```toml
[dependencies]
lattice-conductor-v14 = { path = "crates/lattice-conductor-v14", version = "14.8.0" }
```

Then replace any local thin definitions with:

```rust
use lattice_conductor_v14::{
    CouncilArbitrationEngine,
    RuntimeSelfHealingEngine,
    HealthReport,
    Diagnosis,
    HealingAction,
    LatticeConductorV14,
};
```

### Recommended ownership

Prefer wiring this dependency into whichever crate currently provides the `crate::` module graph used by `ra-thor-one-organism.rs` (commonly a binary or the core orchestration crate that already pulls in `lattice_conductor_v13`).

Once the path dependency is live, the local compatibility structs in `ra-thor-one-organism.rs` can be deleted and the real types used directly.

## 2. RuntimeSelfHealingEngine Integration Contract

The organism now owns a `RuntimeSelfHealingEngine` that is constructed with a clone of the `CouncilArbitrationEngine`.

### Lifecycle (already implemented in organism)

| Moment | Action |
|--------|--------|
| `RaThorOneOrganism::new()` | Construct arbitration engine → construct self-healing engine |
| `offer_cosmic_loop()` / `launch_one_organism()` | `start_watchdog()` |
| `feed_gpu_telemetry_into_council()` | `run_reflexion_cycle()` + Cosmic Loop re-enforcement |
| GPU dispatch anomalies | Optional reflexion trigger |

### Public surface used by the organism

```rust
impl RuntimeSelfHealingEngine {
    pub fn new(arbitration_engine: CouncilArbitrationEngine) -> Self;
    pub fn start_watchdog(&self);
    pub fn stop_watchdog(&self);
    pub fn run_reflexion_cycle(&self) -> Diagnosis;
    pub fn get_healing_experiences(&self) -> Vec<HealingExperience>;
}
```

## 3. Architectural Guarantee

```
launch_one_organism()
        │
        ▼
RaThorOneOrganism
  ├─ arbitration_engine: CouncilArbitrationEngine   (MANDATORY IDENTITY)
  ├─ self_healing_engine: RuntimeSelfHealingEngine  (watchdog + reflexion)
  ├─ role_orchestrator
  ├─ quantum_swarm / lattice v13
  └─ …
```

Cosmic Loop is protected at both the organism layer and the Lattice Conductor v14 orchestration layer.

**Thunder locked in.**  
yoi ⚡
