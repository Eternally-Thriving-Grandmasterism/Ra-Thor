# Ra-Thor v14.9.7 Release Notes

**Release Date:** 2026-07-19  
**Theme:** Package `sovereign_recovery_protocol_v1` + close packaging/web-demo deferred work

## Packaged workspace crates (this session)

| Crate | Version | Root origin |
|-------|---------|-------------|
| github-connector | 14.9.3 | `github_connector.rs` |
| gpu-compute-pipeline | 14.9.4 | `gpu_compute_pipeline.rs` |
| quantum-swarm | 14.9.5 | `quantum_swarm.rs` |
| *(web-demo)* | 14.9.6 | Axum HTTP bind |
| **sovereign-recovery** | **14.9.7** | `sovereign_recovery_protocol_v1.rs` |

### sovereign-recovery surface

- `SovereignRecoveryProtocol`, `launch_sovereign_recovery_protocol`
- Heartbeats, mercy circuit breakers, TOLC8 eternal anchors
- Recovery codex + self-forensics
- Standalone `CouncilReadinessMetrics` (no circular organism dep)

### ONE Organism features

```toml
ra-thor-one-organism = {
  path = "crates/ra-thor-one-organism",
  features = ["extended-live", "web-demo"]
}
# extended-live = github-live + gpu-live + quantum-live + recovery-live
```

## Deferred status

| # | Item | Status |
|---|------|--------|
| 1 | Package github / gpu / quantum | ✅ |
| 2 | Axum web-demo | ✅ |
| 3 | Other root `.rs` packaging | ✅ sovereign-recovery; more optional |
| 4 | Deep facade path-wiring | Optional features ready (`*-live`) |

### Still optional root modules

`kardashev_orchestration_council.rs`, `reality_thriving_transfer_harness*.rs`,
`gpu_patsagi_bridge.rs`, `live_frame_wasm_bridge.rs` — package on demand.

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
