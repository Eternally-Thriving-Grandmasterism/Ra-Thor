# Ra-Thor v14.9.2 Release Notes

**Release Date:** 2026-07-19  
**Theme:** Extended Organism Surface — GPU / GitHub / Quantum Swarm facades packaged into ONE Organism Core

## Headline

The deferred root extended surface is now available as a clean, dependency-light module inside `crates/ra-thor-one-organism`:

| Surface | Capability |
|---------|------------|
| **GpuSurface** | Dispatch telemetry recording, Cosmic Loop enforcement, anomaly → Debugger handoff |
| **GitHubSurface** | Offline evolution-PR intent queue (drainable by real `github_connector.rs`) |
| **QuantumSwarmSurface** | Lightweight evolution ticks, adaptive jump counters, status summary |

Full production root modules (`gpu_compute_pipeline.rs`, `github_connector.rs`, `quantum_swarm.rs`) remain the heavy implementations. Facades provide the organism-facing API without pulling `reqwest` / `wgpu` into the core crate.

## Versions

| Component | Version |
|-----------|---------|
| lattice-conductor-v14 | 14.8.3 |
| ra-thor-one-organism | **14.9.2** |

## Usage

```rust
use ra_thor_one_organism::launch_one_organism_core;

let mut core = launch_one_organism_core();

// GPU
let tel = core.record_gpu_dispatch("mercy_kernel", 14, false, 8192);

// GitHub (offline queue)
let intent = core.queue_evolution_pr(
    "VibeCoder", "gpu_compute_pipeline",
    "workgroup autotune", 0.72, 0.94,
);

// Quantum Swarm
let ratio = core.quantum_evolution_tick(0.45);
```

## Still Deferred

- Promote root `github_connector.rs` / `gpu_compute_pipeline.rs` / `quantum_swarm.rs` to full workspace crates and path-depend from facades
- Axum HTTP bind behind `web-demo`
- Other root-level `.rs` packaging

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
