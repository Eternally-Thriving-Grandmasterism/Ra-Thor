# Ra-Thor v14.9.5 Release Notes

**Release Date:** 2026-07-19  
**Theme:** Promote root `quantum_swarm.rs` → `crates/quantum-swarm@14.9.5`  
**Milestone:** Deferred Item 1 packaging **COMPLETE**

## Packaged extended surfaces

| Crate | Version | Root origin |
|-------|---------|-------------|
| github-connector | 14.9.3 | `github_connector.rs` |
| gpu-compute-pipeline | 14.9.4 | `gpu_compute_pipeline.rs` |
| **quantum-swarm** | **14.9.5** | `quantum_swarm.rs` |

### quantum-swarm surface

- `QuantumSwarmEngine`, `QuantumSwarmConfig`, `QuantumSwarmMember`
- Mean-best tracker, hybrid attractor, adaptive jumps, proposals
- Protected evolution / jump / proposal paths
- Full + async protected benchmark suites
- Optional feature `gpu` → `gpu-compute-pipeline` for real dispatch benches
- Lightweight in-crate `SovereignRecoveryProtocol` (standalone-compilable)

### Organism features

```toml
ra-thor-one-organism = {
  path = "crates/ra-thor-one-organism",
  features = ["extended-live"]  # github + gpu + quantum
}
```

Individual: `github-live` | `gpu-live` | `quantum-live`

Root files are migration shims only.

## Still deferred

2. Axum HTTP bind (`web-demo`)  
3. Other root-level `.rs` packaging  
4. Deep path-wiring of `ExtendedOrganismSurface` facades onto live crates (optional next)

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
