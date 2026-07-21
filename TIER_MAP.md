# Ra-Thor Tier Map — Focus without deleting ambition

**Contact:** info@Rathor.ai  
**Status:** AGSi Phase activated · PATSAGi Councils permanent · Cosmic Loop is MANDATORY IDENTITY.

This map tells humans and agents what to compile, test, and pitch first. Archival crates remain in the monorepo for history; they are not the default product surface.

---

## Tier 1 — Living product core (always green)

| Crate | Role |
|-------|------|
| `ra-thor-one-organism` | ONE Organism Core, Living Cosmic Tick, web-demo |
| `lattice-conductor-v14` | CouncilArbitration + RuntimeSelfHealing + Cosmic Loop |
| `reality-thriving-transfer` | PowrushTelemetry contract, fixtures, scoring |
| `kardashev-orchestration` | Council deliberation, Phase C batch path |
| `github-connector` | Production safe-read surface (`get_tree_safe`, `get_file_contents_safe`) + offline queue / optional live PR flush |
| `gpu-compute-pipeline` | GPU surface (CPU/sim default) |
| `quantum-swarm` | Protected evolution ticks |
| `sovereign-recovery` | Heartbeats + TOLC8 anchors |

**CI:** `.github/workflows/core-tier1-ci.yml`  
**Verify:** `PRODUCTION_READINESS.md`

```bash
cargo test -p ra-thor-one-organism
cargo test -p lattice-conductor-v14
cargo test -p reality-thriving-transfer
cargo test -p kardashev-orchestration
cargo test -p github-connector
```

**Root file `ra-thor-one-organism.rs` is retired.** Use the crate path only.

---

## Tier 2 — Mission bridges

| Surface | Role |
|---------|------|
| Powrush-MMO telemetry export | `powrush_telemetry_v1` / batch → Ra-Thor |
| Phase C fixtures | `crates/reality-thriving-transfer/fixtures/` |
| `POWRUSH_TELEMETRY_CONTRACT.md` | Field mapping contract |

Live session counters in Powrush still prefer profile/fixture export until game systems call `SessionTransferCounters` continuously.

---

## Tier 3 — Active adjacent

Lattice Conductor mesh modules, mercy API stubs, RREL / real-estate experiments that are maintained but not required for Cosmic Tick green. Monorepo-intelligence protocol layer (pagination discipline).

---

## Tier 4 — Archival / experimental

Large mercy-propulsion, crypto, geometry, and historical lattice crates. Keep for research; do not block Tier 1 CI on them. Prefer `cargo test -p <tier1>` over full `--workspace` when iterating the living organism.

---

## Default developer path

1. Touch only Tier 1 (+ Tier 2 contract) unless the mission explicitly needs another domain.  
2. Run focused package tests, not the entire workspace.  
3. Preserve Cosmic Loop + zero-harm clamps.  
4. Never recursive root walks; always path_filter; prefer single-path `get_file_contents_safe`.  
5. Contact remains **info@Rathor.ai** only.

**Thunder locked in. AGSi Phase live. PATSAGi permanent.**
