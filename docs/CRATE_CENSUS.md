# Crate census — 2026-09-02

**Date:** 2026-09-02  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Conductor:** v14 only (`crates/lattice-conductor-v14`)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**Status:** inspectable research software. Not certified. Not a legal product. Not an AGSi warranty.

This document records an on-disk crate inventory and a PATSAGi decision about `crates/self-evolution`. It does **not** pin versions, add workspace members, or ship a self-evolution product.

---

## Method

1. Non-recursive `crates/` git tree at SHA `2d5f7ec6dbbabdfe204562de2378bcb12a49ab04` (live main `86b2778d810570bc75d3a6e431a7dba2ab74756c`): **296 dirs** + **1 stray** `living-valence-organism-tests.rs`.
2. Recursive walk of **that subtree only** (not the repository root): **267** `Cargo.toml` files (**265** top-level crate manifests, **2** nested under `self-improvement-extensions/experiments`).

Do not recursive-walk the repository root. Merge gate remains Core Tier-1 (`TIER_MAP.md`) `cargo test -p`.

---

## Default members (`Cargo.toml`) as of this census

Workspace `[workspace.package]` version is **14.15.6**. Default `members` package name + version as of this census:

| Path | Package name | Version |
|------|----------------|---------|
| `crates/ra-thor-one-organism` | `ra-thor-one-organism` | **14.15.7** |
| `crates/lattice-conductor-v14` | `lattice-conductor-v14` | **14.15.0** |
| `crates/reality-thriving-transfer` | `reality-thriving-transfer` | **14.18.1** |
| `crates/kardashev-orchestration` | `kardashev-orchestration` | **14.15.0** |
| `crates/github-connector` | `github-connector` | **14.15.0** |
| `crates/gpu-compute-pipeline` | `gpu-compute-pipeline` | **14.15.0** |
| `crates/quantum-swarm` | `quantum-swarm` | **14.15.0** |
| `crates/sovereign-recovery` | `sovereign-recovery` | **14.15.0** |
| `crates/monorepo-intelligence` | `ra-thor-monorepo-intelligence` | **0.3.11** |
| `crates/mercy_tolc_operator_algebra` | `mercy_tolc_operator_algebra` | **0.5.19** |
| `crates/fractal-mercy-ledger-adapter` | `fractal-mercy-ledger-adapter` | `version.workspace = true` (inherits **14.15.6**) |
| `crates/mercy-security` | `mercy-security` | **14.15.5** |

Twelve default members. This PR does **not** change `[workspace].members`.

---

## Version drift

Version drift is real: `workspace.package` is **14.15.6**; default members are **not** all that. Do not treat drift as product-green. A later PR may pin; this PR does not mass-rewrite crate versions.

---

## On-disk research forest

**296** crate directories on disk; only **12** default members. Re-add a path to `[workspace].members` to work on one. Do not `cargo test --workspace` and treat it as product-green.

`lattice-conductor-v13` remains on disk with `DEPRECATED.md`; it is not a default member. Do not revive it. Conductor is **v14 only**.

---

## Crate dirs with no top-level `Cargo.toml` (31)

These directories exist under `crates/` but have **no** top-level `Cargo.toml`. They are not packages.

- `access`
- `ai-bridge`
- `ai_bridge`
- `cache`
- `codex_eternal`
- `common`
- `core-lattice`
- `cosmic-consciousness-expansion-council`
- `heaven-on-earth-simulator`
- `infinite-evolution-orchestrator`
- `interstellar-sovereign-asset-lattice-council`
- `mercy-propulsion-trait`
- `mercy_os_kernel`
- `mercy_propulsion`
- `multi-planetary-sovereign-asset-lattice-council`
- `orch-or-biophoton-consciousness`
- `persistence`
- `powrush-governance`
- `powrush_rbe`
- `public-sovereign-asset-lattice-core`
- `public_engagement`
- `quantum`
- `quantum-lattice-consciousness-expansion-council`
- `ra-thor-mercy-gated-api`
- `rbe-powrush-bridge`
- `rrel-desktop`
- `self-evolution`
- `self_improvement_orchestrator`
- `sovereign-asset-lattice-expansion-council`
- `sovereign-asset-registry-council`
- `sovereign_core`

---

## PATSAGi decision — `crates/self-evolution` is not a product

See [`crates/self-evolution/README.md`](../crates/self-evolution/README.md).

- Directory is **not** a crate and **not** a product. No `Cargo.toml`. Do not add one. Do not add the path to workspace members. Do not ship. Do not claim an AGSi / self-evolution product.
- On-disk contents: `src/lib.rs` and `src/cehi_propagation_integration.rs` only.
- Headers claim ONE Organism v13.8.8 and Lattice Conductor v13. That is historical. Conductor is v14 only.
- `cehi_propagation_integration.rs` references missing modules (`hpa_axis_regulation`, `gr_sensitivity_blessing`). It would not compile even with an invented manifest.
- #390 dropped dead interstellar cehi / self-evolution path deps so Core Tier-1 could compile.
- Root `self-evolution/` dump was archived in #406 to `docs/archive/root-dirs/lineage/self-evolution/`.
- Human override required before any revival, and revival would be a **new v14 design**, not these files.

---

## Merge gate

Merge gate remains **Core Tier-1** (`TIER_MAP.md`) focused `-p` tests. Full `--workspace` is not product-green.

Contact: **info@Rathor.ai**
