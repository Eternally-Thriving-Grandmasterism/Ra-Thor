# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

**v0.5.6** — Ambient · Valence · Adaptive floor · Concurrent zones · Soft feedback · LatticeHealthReport · JSON export

## Geometry

| Layer | Dimension | Role |
|-------|-----------|------|
| Ambient space | R^16 | Full action / grief embedding |
| Living Mercy subspace | R^8 | TOLC 8 gates |
| Concurrent zones | N independent | Per-zone basis + staggered Cosmic Ticks |

```
P = E(E^T E)^{-1}E^T
N1(g) = (I - P)g
grief_load = (1 - v) * ||N1(g)||
```

## Surfaces

| Type | Role |
|------|------|
| `SoftFeedbackBridge` | Lattice → experiential surface |
| `SoftFeedbackEvent` / `ZoneSnapshot` | Sealed dual-repo event protocol |
| `LatticeHealthReport` | Machine-readable health (`ra_thor_lattice_health_v1`) |
| `ConcurrentZoneLattice` | Multi-zone stress |

## Soft feedback (dual-repo)

Contract shared with [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO):

```text
SoftFeedbackEvent { zone_id, grief_load, valence, under_floor, tick }
LatticeHealthReport { schema, max_rho, healthy, zones, ... }
```

Powrush orchestrator emits one soft-feedback event per `run_tick` and feeds `soft_feedback_events` / `soft_feedback_total_grief` into telemetry custom metrics.

## Public proofs

```bash
# Human-readable demo
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --agents 12000 --zones 4

# Machine-readable JSON (CI / Powrush consumers)
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --agents 3000 --zones 3 --json

# High-grief stress
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench -- --agents 50000 --zones 5

cargo test -p mercy_tolc_operator_algebra
```

19 property tests. Demo exits non-zero on gate failure.

## License

AG-SML v1.0 — Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
