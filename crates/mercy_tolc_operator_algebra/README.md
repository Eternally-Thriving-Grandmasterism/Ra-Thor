# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

**v0.5.7** — Ambient · Valence · Adaptive floor · Concurrent zones · Soft feedback · LatticeHealthReport · Adaptive Cosmic Tick

See [LATTICE_STATUS.md](./LATTICE_STATUS.md) for the full layer review.

## Geometry

| Layer | Dimension | Role |
|-------|-----------|------|
| Ambient space | R^16 | Full action / grief embedding |
| Living Mercy subspace | R^8 | TOLC 8 gates |
| Concurrent zones | N independent | Per-zone basis + adaptive Cosmic Ticks |

```
P = E(E^T E)^{-1}E^T
N1(g) = (I - P)g
grief_load = (1 - v) * ||N1(g)||
effective_period(z) = max(min_period, base / (1 + grief_z / scale))
```

## Surfaces

| Type | Role |
|------|------|
| `SoftFeedbackBridge` | Lattice → experiential surface |
| `LatticeHealthReport` | Machine-readable health (`ra_thor_lattice_health_v1`) |
| `ConcurrentZoneLattice` | Multi-zone stress + adaptive purify |

## Dual-repo

Contract shared with [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO):

```text
SoftFeedbackEvent { zone_id, grief_load, valence, under_floor, tick }
```

## Public proofs

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --json
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench -- --agents 50000 --zones 5
```

20 property tests.

## License

AG-SML v1.0 — Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
