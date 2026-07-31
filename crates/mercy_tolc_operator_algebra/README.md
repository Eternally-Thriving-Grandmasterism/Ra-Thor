# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

**v0.5.4** — Ambient elevation · Valence · Adaptive floor · Concurrent zones · Soft feedback bridge

## Geometry

| Layer | Dimension | Role |
|-------|-----------|------|
| Ambient space | R^16 | Full action / grief embedding |
| Living Mercy subspace | R^8 | TOLC 8 gates |
| Orthogonal complement | R^8 | Pure grief (coords 8..15) |
| Concurrent zones | N independent | Per-zone basis + staggered Cosmic Ticks |

```
P = E(E^T E)^{-1}E^T
N1(g) = (I - P)g
grief_load = (1 - v) * ||N1(g)||
floor(v) = MERCY_PURITY_FLOOR * (1 + 99*(1-v))
```

## Surfaces

| Type | Role |
|------|------|
| `LivingMercyBasis` / `MercyProjector` / `NilpotentSuppressor` | Core algebra |
| `Valence` | Grief intensity + adaptive floor |
| `ZoneState` / `ConcurrentZoneLattice` | Multi-zone stress |
| `SoftFeedbackBridge` / `SoftFeedbackEvent` / `ZoneSnapshot` | Dual-repo sealed protocol |
| `ModifiedGramSchmidt` | Cosmic Tick purification |

## Soft feedback (dual-repo)

Sealed event contract shared with [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO) `ra_thor_bridge`:

```text
SoftFeedbackEvent { zone_id, grief_load, valence, under_floor, tick }
ZoneSnapshot      { zone_id, grief_absorbed, vectors_processed, last_rho }
```

Ra-Thor side: `SoftFeedbackBridge::ingest` / `drain_events` / `snapshots`  
Powrush side: `RaThorBridge::report_zone_grief` / `drain_soft_feedback` / `soft_zone_snapshots`

No hard crate dependency either way — serde field shape is the contract.

## Benchmark

```bash
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench -- --agents 50000 --zones 5
cargo test -p mercy_tolc_operator_algebra
```

18 property tests. Five verification gates on the public harness.

## License

AG-SML v1.0 — Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
