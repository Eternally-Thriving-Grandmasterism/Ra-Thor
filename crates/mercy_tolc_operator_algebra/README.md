# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

**v0.5.11** — Ambient · Valence · Adaptive floor · Concurrent zones · Soft feedback · LatticeHealthReport · Adaptive Cosmic Tick · Stress EMA · Health aggregates · Telemetry surface

See [LATTICE_STATUS.md](./LATTICE_STATUS.md) for the full layer review.

## Geometry

```
P = E(E^T E)^{-1}E^T
N1(g) = (I - P)g
grief_load = (1 - v) * ||N1(g)||
stress_ema ← (1−α)·stress_ema + α·load
effective_period(z) = max(min_period, base / (1 + stress_ema / scale))
```

## Dual-repo

Contract shared with [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO):

```text
SoftFeedbackEvent { zone_id, grief_load, valence, under_floor, tick }
ZoneSnapshot { + stress_ema, purify_count, effective_period }
LatticeHealthReport { + total_purify_count, max_stress_ema, mean_effective_period }
```

## Public proofs

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --agents 3000 --zones 3 --json
```

24 property tests.

## License

AG-SML v1.0 — Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
