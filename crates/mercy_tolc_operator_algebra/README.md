# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

**v0.5.13** — Ambient · Valence · Adaptive floor · Concurrent zones · Soft feedback · LatticeHealthReport · Adaptive Cosmic Tick · Stress EMA · Health aggregates · Telemetry · Composite health_score · Score gate

See [LATTICE_STATUS.md](./LATTICE_STATUS.md) and [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

## Geometry

```
P = E(E^T E)^{-1}E^T
N1(g) = (I - P)g
grief_load = (1 - v) * ||N1(g)||
stress_ema ← (1−α)·stress_ema + α·load
health_score = purity_term × stress_term ∈ [0, 1]
```

## Dual-repo

```text
SoftFeedbackEvent { zone_id, grief_load, valence, under_floor, tick }
ZoneSnapshot { + stress_ema, purify_count, effective_period }
LatticeHealthReport { + health_score }
CI: healthy && health_score ≥ 0.5
```

## Public proofs

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --agents 3000 --zones 3 --json
```

25 property tests.

## License

AG-SML v1.0 — Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
