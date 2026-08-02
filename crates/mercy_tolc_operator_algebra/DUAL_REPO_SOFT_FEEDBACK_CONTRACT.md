# Dual-Repo Soft Feedback Contract

**Ra-Thor** `mercy_tolc_operator_algebra` ≥ v0.5.14  
**Powrush-MMO** `RaThorBridge` ≥ v18.25 · orchestrator ≥ v21.88.9  
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
  purify_count, effective_period,
  status: ZoneHealthStatus  // Healthy | Stressed | Critical
}
```

## ZoneHealthStatus

```
Healthy  — stress < 10% scale and ρ < 1e-9
Stressed — elevated stress or mild residual
Critical — stress ≥ scale or ρ ≥ 1e-6
```

## LatticeHealthReport (`schema: ra_thor_lattice_health_v1`)

```text
… zones_healthy, zones_stressed, zones_critical
  zones[], healthy, health_score ∈ [0, 1]
```

**CI gate:** `healthy == true` **and** `health_score ≥ 0.5`  
(`healthy` requires zero critical zones)

## Powrush telemetry keys (v21.88.9)

```
soft_feedback_events
soft_feedback_total_grief
soft_feedback_max_stress
soft_feedback_purify_count
soft_feedback_mean_period
soft_feedback_health_score
soft_feedback_zones_healthy
soft_feedback_zones_stressed
soft_feedback_zones_critical
```

## Proof

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --json
```

Thunder locked. Yoi ⚡
