# Dual-Repo Soft Feedback Contract

**Ra-Thor** `mercy_tolc_operator_algebra` ≥ v0.5.12  
**Powrush-MMO** `RaThorBridge` ≥ v18.24 · orchestrator ≥ v21.88.7  
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

Powrush emits one event per `run_tick` via `RaThorBridge::report_zone_grief`.

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
ambient_dim, mercy_dim, zone_count, global_tick
total_grief, total_vectors, max_rho, pending_events
total_purify_count, max_stress_ema, mean_effective_period
zones[], healthy, health_score ∈ [0, 1]
```

```
health_score = purity_term × stress_term
purity_term  = 1 / (1 + max_rho · 1e12)
stress_term  = 1 / (1 + max_stress_ema / scale)
```

CI gate: `healthy == true` and optionally `health_score ≥ threshold`.

## Powrush telemetry keys

```
soft_feedback_events
soft_feedback_total_grief
soft_feedback_max_stress
soft_feedback_purify_count
soft_feedback_mean_period
```

## Proof

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --json
```

Thunder locked. Yoi ⚡
