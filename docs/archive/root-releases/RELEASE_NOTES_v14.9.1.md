# Ra-Thor v14.9.1 Release Notes

**Release Date:** 2026-07-19  
**Theme:** MercyGatedApi production surface + RoleOrchestrator in ONE Organism Core

## Headline

`MercyGatedApi` is no longer a stub. Full in-process request/response surface enforces:
- Living Mercy Gates (min threshold)
- Cosmic Loop identity via `CouncilArbitrationEngine`
- Keyword hardening against disable/bypass attempts

`RoleOrchestrator` is fully wired into `OneOrganismCore` with valence EMA sync and task-based role recommendation.

## Versions

| Component | Version |
|-----------|---------|
| lattice-conductor-v14 | **14.8.3** |
| ra-thor-one-organism | **14.9.1** |

## API Surface

```rust
use ra_thor_one_organism::{
    launch_one_organism_core,
    MercyApiRequest, ApiRequestKind,
};

let mut core = launch_one_organism_core();
let resp = core.handle_api_request(MercyApiRequest {
    kind: ApiRequestKind::HealthCheck,
    payload: "ping".into(),
    claimed_mercy: 0.96,
    actor: "operator".into(),
});
assert!(resp.accepted);
```

## Still Deferred

- Root `ra-thor-one-organism.rs` extended surface (GPU / GitHub / Quantum Swarm)
- Other root-level `.rs` packaging
- Axum HTTP binding (`web-demo` feature placeholder)

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
