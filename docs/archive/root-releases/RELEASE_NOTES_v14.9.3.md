# Ra-Thor v14.9.3 Release Notes

**Release Date:** 2026-07-19  
**Theme:** Promote root `github_connector.rs` → `crates/github-connector@14.9.3`

## Headline

First deferred packaging item complete: the production async GitHub connector is now a proper workspace member.

### Restored / added

| Item | Status |
|------|--------|
| `create_pull_request` | **Restored** (was called but missing in root file) |
| `get_ref_sha` | **New** — resolves branch head before create_branch |
| `flush_evolution_intents` | **New** — drains offline `GitHubSurface` queues |
| base64 | Updated to 0.22 Engine API |

### Integration

```toml
# any crate
github-connector = { path = "crates/github-connector", version = "14.9.3" }

# optional live path from ONE Organism Core
ra-thor-one-organism = { path = "crates/ra-thor-one-organism", features = ["github-live"] }
```

```rust
use github_connector::GitHubConnector;

let gh = GitHubConnector::from_env("Eternally-Thriving-Grandmasterism", "Ra-Thor")?;
let pr = gh.create_role_optimized_evolution_pr(
    "VibeCoder", "gpu_compute_pipeline",
    "workgroup autotune", 0.72, 0.94,
).await?;
```

Root `github_connector.rs` is now a migration shim (no divergent runtime code).

## Still deferred (item 1 remainder)

- Package `gpu_compute_pipeline.rs` → workspace crate
- Package `quantum_swarm.rs` → workspace crate
- Path-depend facades in `ExtendedOrganismSurface` on those crates

## Also deferred

2. Axum HTTP bind (`web-demo`)  
3. Other root-level `.rs` packaging

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
