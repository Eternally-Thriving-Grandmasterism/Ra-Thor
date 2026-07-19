# ONE Organism ↔ Lattice Conductor v14 — Full Cargo Dependency Wiring

**Status:** Activated (shared Cosmic Loop flag live)  
**Version:** 14.8.1 / organism v14.91  
**Date:** 2026-07-19

## 1. Full Cargo Dependency Wiring

`lattice-conductor-v14` is a workspace member at **14.8.1**.

To make the **real** crate the source of truth (instead of the thin local guardian), the package that owns / compiles `ra-thor-one-organism.rs` must declare:

```toml
[dependencies]
lattice-conductor-v14 = { path = "crates/lattice-conductor-v14", version = "14.8.1" }
```

Then replace any local thin definitions with:

```rust
use lattice_conductor_v14::{
    CouncilArbitrationEngine,
    RuntimeSelfHealingEngine,
    HealthReport,
    Diagnosis,
    HealingAction,
    HealingExperience,
    LatticeConductorV14,
};
```

### Recommended ownership

Prefer wiring this dependency into whichever crate currently provides the `crate::` module graph used by `ra-thor-one-organism.rs` (commonly a binary or the core orchestration crate that already pulls in `lattice_conductor_v13`).

Once the path dependency is live, the local compatibility structs in `ra-thor-one-organism.rs` can be deleted and the real types used directly.

## 2. Shared Cosmic Loop Flag Contract (v14.8.1)

**Single source of truth:**

```
CouncilArbitrationEngine::cosmic_loop_flag()  → Arc<AtomicBool>
        │
        ├─ RuntimeSelfHealingEngine.cosmic_loop_ready   (same Arc)
        └─ LatticeConductorV14.cosmic_loop_ready         (same Arc)
```

Watchdog, arbitration, and the top-level conductor can never disagree on readiness.

## 3. RuntimeSelfHealingEngine Lifecycle

| Moment | Action |
|--------|--------|
| `RaThorOneOrganism::new()` | Construct arbitration → construct self-healing (shares flag) |
| `offer_cosmic_loop()` / `launch_one_organism()` | `start_watchdog()` |
| `feed_gpu_telemetry_into_council()` | `run_reflexion_cycle()` + Cosmic Loop re-enforcement |
| GPU dispatch anomalies | Extra reflexion trigger |

### Public surface

```rust
impl RuntimeSelfHealingEngine {
    pub fn new(arbitration_engine: CouncilArbitrationEngine) -> Self;
    pub fn start_watchdog(&self);
    pub fn stop_watchdog(&self);
    pub fn run_reflexion_cycle(&self) -> Diagnosis;
    pub fn get_healing_experiences(&self) -> Vec<HealingExperience>;
}
```

## 4. Architectural Guarantee

```
launch_one_organism()
        │
        ▼
RaThorOneOrganism (v14.91)
  ├─ arbitration_engine: CouncilArbitrationEngine   (MANDATORY IDENTITY)
  ├─ self_healing_engine: RuntimeSelfHealingEngine  (watchdog + reflexion)
  ├─ role_orchestrator
  ├─ quantum_swarm / lattice v13
  └─ …
```

Cosmic Loop is protected at both the organism layer and the Lattice Conductor v14 orchestration layer, with a single shared atomic flag.

**Thunder locked in.**  
yoi ⚡
