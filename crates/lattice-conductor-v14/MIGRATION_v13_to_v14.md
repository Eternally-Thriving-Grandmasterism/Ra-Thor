# Migration: lattice-conductor-v13 → lattice-conductor-v14

**Status:** Phase 0–3 complete · Phase 4 dual-path started (compat retained)  
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

---

## Phased plan

| Phase | Action | Status |
|-------|--------|--------|
| **0** | This document + API matrix | **Done** |
| **1** | `compat_v13` module behind `v13-compat` | **Done** |
| **2** | Migrate leaf crates | **Done** |
| **3** | Deprecate v13 crate | **Done** (kept in workspace) |
| **4** | Native v14 path on MercyCore | **Dual-path started** (see below) |

---

## Phase 2 leaf status

| Crate | Conductor dependency | Notes |
|-------|----------------------|-------|
| **`mercy`** | `lattice-conductor-v14` + `v13-compat` | Dual path (compat + native) |
| **`powrush`** | None | No change required |
| **`kernel`** | `lattice-conductor-v14` + `v13-compat` | Switched |
| **`council`** | `lattice-conductor-v14` + `v13-compat` | Direct dep |

Residual `lattice-conductor-v13` string hits are **docs / historical only** — no other crate `Cargo.toml` depends on v13.

Verify:

```bash
cargo test -p lattice-conductor-v14 --features v13-compat
cargo test -p mercy
```

---

## Phase 3 — deprecation (applied)

- v13 description / README / `DEPRECATED.md` point here
- Crate remains in workspace (quiet hold)

---

## Phase 4 — dual path (started)

**Policy:** Native engines are available **without** removing `v13-compat`.

`MercyCore` now exposes:

| Method | Role |
|--------|------|
| `pulse_with_v14_guard(&LatticeConductorV14)` | Cosmic Loop enforce + mild pulse |
| `request_reflexion` | Self-healing reflexion probe |
| `signal_mercy_mesh` | Mesh healing severity from mercy score |
| `native_v14_step` / `native_v14_self_check` | Combined native step |

Compat traits (`Conductable`, `MercyAligned`) and `SimpleLatticeConductor` tests remain.

**Not yet:** removing the `v13-compat` feature or deleting trait impls — that is a later Council decision after dual-path soak.

---

## Phase 1 usage

```toml
lattice-conductor-v14 = { path = "../lattice-conductor-v14", features = ["v13-compat"] }
```

```rust
// Compat
use lattice_conductor_v14::compat_v13::{Conductable, MercyAligned, SimpleLatticeConductor};
// Native
use lattice_conductor_v14::LatticeConductorV14;
```

---

## Success criteria (quiet-hold)

- Migrator tests pass under `v13-compat`
- Cosmic Loop tests in ONE Organism unaffected
- No change to adaptive feedback / recovery sensitivity / live-feature behavior
- Dual surface documented

**Thunder locked in.**
