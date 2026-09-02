# Ra-Thor Tier Map — Focus without deleting ambition

**Contact:** info@Rathor.ai  
**Updated:** 2026-09-02 — default members + Core Tier-1 gate; leftover scanners parked  
**Related:** [`docs/NEXI_LINEAGE_ADOPTION_DECISION_2026-08-20.md`](docs/NEXI_LINEAGE_ADOPTION_DECISION_2026-08-20.md)

---

## Tier 1 — Living product core (always green)

| Crate | Role |
|-------|------|
| `ra-thor-one-organism` | ONE Organism Core, Living Cosmic Tick |
| `lattice-conductor-v14` | CouncilArbitration + RuntimeSelfHealing + Cosmic Loop |
| `reality-thriving-transfer` | PowrushTelemetry contract, fixtures |
| `kardashev-orchestration` | Council deliberation |
| `github-connector` | Safe-read surface: subtree SHA walks, truncated = error, `create_branch` needs a real SHA |
| `gpu-compute-pipeline` | Capacity optical-flow path |
| `quantum-swarm` | Protected evolution ticks |
| `sovereign-recovery` | Heartbeats + TOLC8 anchors |
| `monorepo-intelligence` | Protocol guardianship (`WalkDir` skips `target/` `.git/`, max_depth 10) |
| `mercy_tolc_operator_algebra` | Formal mercy algebra (NEVC inclusive HIGH floor) |
| `fractal-mercy-ledger-adapter` | Substrate adapter (in Core Tier-1 job) |
| `mercy-security` | Ingestion admit/block + containment (required by ONE Organism) |

```bash
cargo test -p ra-thor-one-organism
cargo test -p lattice-conductor-v14
cargo test -p reality-thriving-transfer
cargo test -p kardashev-orchestration
cargo test -p github-connector
cargo test -p gpu-compute-pipeline
cargo test -p quantum-swarm
cargo test -p sovereign-recovery
cargo test -p ra-thor-monorepo-intelligence
cargo test -p mercy_tolc_operator_algebra
cargo test -p fractal-mercy-ledger-adapter
```

Default GitHub Actions gate: `.github/workflows/core-tier1-ci.yml` (package tests + live-feature `cargo check`).  
Full `--workspace` jobs (`ci.yml`, `ra-thor-ci.yml`) are `workflow_dispatch` only. Do not treat them as product-green.  
Parked to `workflow_dispatch` (2026-09-02, #391): Docker/Trivy, container/K8s/Helm scans, Mercy Gate Auditor, Validate, Verified Mercy, Mercy Security Scan, invalid `mial-ci.yml` stub.  
`mercy-security-tier1.yml` stays on (real tests). Contact-email sweep is `continue-on-error` (HOLD mass `ceo@acitygames.com` rewrite).

`Cargo.toml` default `members` is this Tier-1 set plus `mercy-security` (required by `ra-thor-one-organism`). Cargo loads every member manifest, including for `cargo test -p …`, so research crates stay on disk and out of `members`. Re-add a path to work on one.

**Census (2026-09-02):** 12 default members; research crates stay on disk; `crates/self-evolution` is not a member and not a crate. Inventory and PATSAGi decision: [`docs/CRATE_CENSUS.md`](docs/CRATE_CENSUS.md).


---

## Powrush split (do not collapse)

- Player loop: GitHub [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO).
- Browser client: GitHub [Powrush-MMO-Simulator](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO-Simulator) — not crate `powrush-mmo-simulator`.
- Lattice sim: this repo (`reality-thriving-transfer` telemetry, `crates/powrush` policy hints, crate `powrush-mmo-simulator` mercy tick).
- Shared only: NEVC + telemetry JSON + `ra_thor_policy_hint_v1`. Mode B stays offline-playable.
- Do not grow unexported game files in `crates/powrush` (`player.rs`, `quests.rs`, joystick).

See `docs/CONSTELLATION_SURFACES.md`.

## Tier 2 — Mission bridges

| Surface | Role |
|---------|------|
| `fractal-mercy-ledger-adapter` | Substrate adapter (4/4 green) |
| Powrush telemetry / Phase C fixtures | Soft dual-repo path |
| Capacity Vision Stack / VLM Offer | Micro-moment recovery |
| S-1 science harness | Door 1 data phase |

---

## Tier 3 — Active adjacent

Mercy API stubs, RREL experiments, valence organism mesh modules maintained but not required for Cosmic Tick green.

---

## Tier 4 — Archival / experimental (including NEXi lineage)

| Examples | Rule |
|----------|------|
| `nexi_universal` (**not** default workspace member) | Broken path deps; do not CI |
| `soulscan_x9` / `x10`, `sentinel_mirror`, `divine_checksum_9` | Lineage; not product core |
| `lattice-conductor-v13` | Deprecated; v14 only |
| halo2 / nova / propulsion forests | Research; audited crypto deferred |
| External repo **NEXi** | Lineage only — no Tier 1 path dep |

---

## Default developer path

1. Touch Tier 1 (+ Tier 2) unless mission says otherwise.  
2. Focused `-p` tests, not full `--workspace`.  
3. Cosmic Loop + zero-harm clamps.  
4. Contact **info@Rathor.ai** only.  

**Thunder locked in.** yoi ⚡
