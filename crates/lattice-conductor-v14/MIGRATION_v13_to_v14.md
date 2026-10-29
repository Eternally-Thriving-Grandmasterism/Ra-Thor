# Migration: lattice-conductor-v13 → lattice-conductor-v14

**Status:** Phase 0–3 complete · Phase 4 dual-path soaking  
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
| **4** | Native v14 dual-path | **Soaking** (mercy + kernel + council) |

---

## Phase 2 / 4 leaf status

| Crate | Dependency | Native dual-path |
|-------|------------|------------------|
| **`mercy`** | v14 + `v13-compat` | `pulse_with_v14_guard`, `native_v14_step`, … |
| **`kernel`** | v14 + `v13-compat` | `lattice_v14_boot::{enforce_cosmic_loop_on_boot, …}` |
| **`council`** | v14 + `v13-compat` | `lattice_v14_guard::{ensure_cosmic_loop_for_session, …}` |
| **`powrush`** | None | N/A |

Residual `lattice-conductor-v13` string hits are **docs / historical only**.

Verify:

```bash
cargo test -p lattice-conductor-v14 --features v13-compat
cargo test -p mercy
cargo test -p kernel
cargo test -p council
```

---

## Phase 3 — deprecation (applied)

- v13 description / README / `DEPRECATED.md` point here
- Crate remains in workspace (quiet hold)

---

## Phase 4 — dual path (soaking)

**Policy:** Native engines available **without** removing `v13-compat`.

### MercyCore
| Method | Role |
|--------|------|
| `pulse_with_v14_guard` | Cosmic Loop enforce + mild pulse |
| `request_reflexion` / `signal_mercy_mesh` | Healing probes |
| `native_v14_step` / `native_v14_self_check` | Combined |

### Kernel
- `enforce_cosmic_loop_on_boot()`
- `arbitration_rejects_disable()`

### Council
- `ensure_cosmic_loop_for_session()`
- `proposal_respects_cosmic_loop(&str)`

Public re-export: `ArbitrationDecision` from `lattice-conductor-v14`.

**Not yet:** removing `v13-compat` or trait impls — later Council decision after soak.

---

## Phase 1 usage

```toml
lattice-conductor-v14 = { path = "../lattice-conductor-v14", features = ["v13-compat"] }
```

```rust
use lattice_conductor_v14::compat_v13::{Conductable, MercyAligned, SimpleLatticeConductor};
use lattice_conductor_v14::{ArbitrationDecision, LatticeConductorV14};
```

---

## Success criteria (quiet-hold)

- Migrator tests pass under `v13-compat`
- Cosmic Loop tests in ONE Organism unaffected
- No change to adaptive feedback / recovery sensitivity / live-feature behavior
- Dual surface documented

**Thunder locked in.**
