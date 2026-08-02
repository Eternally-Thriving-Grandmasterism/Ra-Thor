# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

**v0.5.18** — Ambient ℝ¹⁶ ⊃ Mercy ℝ⁸ · Valence · Adaptive floor · Concurrent zones · Soft feedback · LatticeHealthReport · Adaptive Cosmic Tick · Stress EMA · ZoneHealthStatus · Critical auto-remediate · Soft-remediate Stressed · Valence histogram · Grief-rate metrics

See [LATTICE_STATUS.md](./LATTICE_STATUS.md) and [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

## Geometry

```
P = E(EᵀE)⁻¹Eᵀ
N₁(g) = (I − P)g
grief_load = (1 − v)·‖N₁(g)‖
stress_ema ← (1−α)·stress_ema + α·load
health_score = purity_term × stress_term ∈ [0, 1]
```

## Remediation tiers

```
Critical → force purify() + critical_auto_purify_count
Stressed → decay_stress(α) + soft_remediate_count
```

## Dual-repo sealed protocol

```text
SoftFeedbackEvent { zone_id, grief_load, valence, under_floor, tick }
ZoneSnapshot      { + stress_ema, purify_count, status, soft_remediate_count }
LatticeHealthReport {
  health_score, valence_mercy_ratio,
  grief_per_tick, vectors_per_tick,
  soft_remediate_rate, critical_auto_rate
}
CI: healthy && health_score ≥ 0.5
```

## Public proofs

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --agents 3000 --zones 3 --json
```

35 property tests.

## License

AG-SML v1.0 — Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
