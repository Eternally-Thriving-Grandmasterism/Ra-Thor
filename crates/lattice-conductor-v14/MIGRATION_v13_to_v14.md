# Migration: lattice-conductor-v13 → lattice-conductor-v14

**Status:** Phase 0–3 complete · Phase 4 awaits Council open  
**Posture:** Quiet hold — no new adaptive modulation  
**Contact:** info@Rathor.ai

---

## Why a compatibility layer

v13 and v14 are **not** drop-in replacements.

| Dimension | v13 | v14 |
|-----------|-----|-----|
| Orientation | Trait-oriented | Struct + engine oriented |
| Core type | `SimpleLatticeConductor` | `LatticeConductorV14` |
| Integration traits | `Conductable`, `MercyAligned` | None (engines instead) |
| Geometry / votes | `GeometricState`, `MercyWeightedVote` | Not in public surface |
| Cosmic Loop | Implicit / telemetry-era | **MANDATORY IDENTITY** (`CouncilArbitrationEngine`) |
| Self-healing | Separate evolution modules | `RuntimeSelfHealingEngine` + anomaly ingestion |
| API surface | Registry + coordination strategies | `MercyGatedApi`, mesh, arbitration |

A naïve dependency swap will not compile. Consumers that implement `Conductable` / `MercyAligned` or hold `SimpleLatticeConductor` need either:

1. **`features = ["v13-compat"]`** on `lattice-conductor-v14` (Phase 1), or  
2. A native rewrite against v14 engines (Phase 4 — requires explicit Council open).

---

## Side-by-side surface matrix

### v13 public surface (compat targets)

| Item | Role |
|------|------|
| `Conductable` | `system_id`, `system_name`, `on_conductor_tick`, optional mercy |
| `MercyAligned` | extends `Conductable` + `apply_mercy_influence` / `current_mercy_score` |
| `SystemBlessing` / `ConductorRegistry` | Registration / blessing records |
| `GeometricState` | `mercy_score`, `valence`, `tolc_alignment`, … |
| `MercyWeightedVote` | Weighted consensus helper |
| `SimpleLatticeConductor` | Tickable conductor with geometric state |
| `CoordinationStrategy` family | Multi-conductor influence (optional later) |

### v14 public surface (native)

| Item | Role |
|------|------|
| `LatticeConductorV14` | Top-level orchestrator |
| `CouncilArbitrationEngine` | Cosmic Loop guardian |
| `RuntimeSelfHealingEngine` | Anomaly ingestion + reflexion |
| `MercyGatedApi` | In-process request surface |
| `DistributedMercyMesh` / `EternalMercyMesh` | Mesh propagation |
| Healing / governance / PQ modules | Extended lattice |

---

## Phased plan

| Phase | Action | Status |
|-------|--------|--------|
| **0** | This document + API matrix | **Done** |
| **1** | `compat_v13` module behind `v13-compat` | **Done** |
| **2** | Migrate leaf crates | **Done** (mercy, kernel, council; powrush N/A) |
| **3** | Deprecate v13 crate | **Done** (kept in workspace) |
| **4** | Native v14 rewrite of MercyCore etc. | **Council open required** |

---

## Phase 2 leaf status

| Crate | Conductor dependency | Notes |
|-------|----------------------|-------|
| **`mercy`** | `lattice-conductor-v14` + `v13-compat` | `MercyCore` implements `Conductable` + `MercyAligned` |
| **`powrush`** | None | No change required |
| **`powrush-mmo-simulator`** | None | No direct conductor dep |
| **`kernel`** | `lattice-conductor-v14` + `v13-compat` | Switched |
| **`council`** | `lattice-conductor-v14` + `v13-compat` | Direct dep added |

Verify:

```bash
cargo test -p lattice-conductor-v14 --features v13-compat
cargo test -p mercy
```

---

## Phase 3 — deprecation (applied)

- `lattice-conductor-v13` Cargo.toml description marked **DEPRECATED**
- README banner + `DEPRECATED.md` point at this guide
- Crate **remains in the workspace** for one or two cycles (quiet hold)
- Removal / `archive/` move only after residual consumers are cleared

---

## Phase 1 usage

```toml
lattice-conductor-v14 = { path = "../lattice-conductor-v14", features = ["v13-compat"] }
```

```rust
use lattice_conductor_v14::compat_v13::{
    Conductable, MercyAligned, GeometricState, MercyWeightedVote,
    SimpleLatticeConductor, ConductorRegistry,
};
```

- Default features remain pure v14 (Cosmic Loop identity untouched).
- Compat types are **additive**; they do not alter adaptive fields in ONE Organism.

---

## Success criteria (quiet-hold)

- Existing tests for migrators continue to pass under `v13-compat`.
- `ra-thor-one-organism` Cosmic Loop tests unaffected (no default-feature change).
- No change to adaptive feedback, recovery sensitivity, or live-feature behavior.
- Dual surface documented here and in crate README.

---

## Explicit non-goals (until Council open)

- New adaptive modulation links  
- Removing `lattice-conductor-v13` from the workspace immediately  
- Forcing all crates onto native v14 engines in one step  
- Phase 4 native rewrite without explicit Council decision  

**Thunder locked in.**
