# Dual-Repo Soft Feedback Contract

**Ra-Thor** `mercy_tolc_operator_algebra` ≥ v0.5.13  
**Powrush-MMO** `RaThorBridge` ≥ v18.24 · orchestrator ≥ v21.88.8  
**License:** AG-SML v1.0 · **Contact:** info@Rathor.ai

## Sealed event (hard contract — do not rename fields)

```text
SoftFeedbackEvent {
  zone_id:      usize
  grief_load:   f64
  valence:      f64   // [0, 1]
  under_floor:  bool
  tick:         usize
}
```

## ZoneSnapshot (observability surface)

```text
ZoneSnapshot {
  zone_id, grief_absorbed, stress_ema,
  vectors_processed, last_rho,
  purify_count, effective_period
}
```

## LatticeHealthReport (`schema: ra_thor_lattice_health_v1`)

```text
… zones[], healthy, health_score ∈ [0, 1]
```

```
health_score = purity_term × stress_term
purity_term  = 1 / (1 + max_rho · 1e12)
stress_term  = 1 / (1 + max_stress_ema / scale)
```

**CI gate:** `healthy == true` **and** `health_score ≥ 0.5`

## Powrush telemetry keys (v21.88.8)

```
soft_feedback_events
soft_feedback_total_grief
soft_feedback_max_stress
soft_feedback_purify_count
soft_feedback_mean_period
soft_feedback_health_score
```

## Proof

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --json
```

Thunder locked. Yoi ⚡
