# Ra-Thor v14.9.4 Release Notes

**Release Date:** 2026-07-19  
**Theme:** Promote root `gpu_compute_pipeline.rs` → `crates/gpu-compute-pipeline@14.9.4`

## Headline

Second packaging slice of deferred item 1 complete.

| Crate | Version | Notes |
|-------|---------|-------|
| github-connector | 14.9.3 | Live API |
| **gpu-compute-pipeline** | **14.9.4** | Public API + CPU/sim path |

### Public surface preserved

`GpuTask`, `GpuTaskResult`, `GpuComputePipeline`, `LumaRing`, `LumaFrame`,
`MotionFieldSoA`, `CommonFateResult`, `dispatch_gpu_task`, pyramid / BM /
Common Fate perception path (CPU simulation).

### Features

```toml
gpu-compute-pipeline = { path = "crates/gpu-compute-pipeline", version = "14.9.4" }
# optional real device (WGSL via monorepo shaders/):
# features = ["wgpu"]

# from ONE Organism:
ra-thor-one-organism = { path = "crates/ra-thor-one-organism", features = ["gpu-live"] }
# or both:
# features = ["extended-live"]  # github-live + gpu-live
```

Root `gpu_compute_pipeline.rs` is now a migration shim.

## Item 1 remainder

- Package `quantum_swarm.rs` → workspace crate
- Path-depend `ExtendedOrganismSurface` facades on packaged crates

## Also deferred

2. Axum HTTP bind (`web-demo`)  
3. Other root-level `.rs` packaging

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
