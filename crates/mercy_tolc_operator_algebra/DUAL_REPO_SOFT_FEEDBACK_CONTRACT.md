# Dual-Repo Soft Feedback Contract

**Ra-Thor** `mercy_tolc_operator_algebra` ≥ v0.5.18  
**Powrush-MMO** `RaThorBridge` ≥ v18.29 · orchestrator ≥ v21.88.13  
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
  status: ZoneHealthStatus,          // Healthy | Stressed | Critical
  critical_auto_purify_count: usize  // priority Cosmic Ticks
  soft_remediate_count: usize        // soft cooling cycles
}
```

## ZoneHealthStatus

```
Healthy  — stress < 10% scale and ρ < 1e-9
Stressed — elevated stress or mild residual
Critical — stress ≥ scale or ρ ≥ 1e-6
```

## Remediation tiers

```
Critical → force purify() + critical_auto_purify_count
Stressed → decay_stress(α) + soft_remediate_count
```

## Valence histogram

```
HIGH  — valence ≥ 0.999999  (mercy soft path)
MID   — valence ≥ 0.5
LOW   — valence < 0.5
mercy_ratio = high / (high + mid + low)
```

## Rate metrics (v0.5.18)

```
grief_per_tick      = total_grief / global_tick
vectors_per_tick    = total_vectors / global_tick
soft_remediate_rate = soft_remediates / global_tick
critical_auto_rate  = critical_auto_purifies / global_tick
```

## LatticeHealthReport (`schema: ra_thor_lattice_health_v1`)

```text
… critical_auto_purifies, soft_remediates,
  valence_high/mid/low_count, valence_mercy_ratio,
  grief_per_tick, vectors_per_tick,
  soft_remediate_rate, critical_auto_rate,
  zones[], healthy, health_score ∈ [0, 1]
```

**CI gate:** `healthy == true` **and** `health_score ≥ 0.5`

## Powrush telemetry keys (v21.88.13)

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
soft_feedback_critical_auto
soft_feedback_soft_remediates
soft_feedback_valence_high
soft_feedback_valence_mid
soft_feedback_valence_low
soft_feedback_mercy_ratio
soft_feedback_grief_per_tick
soft_feedback_vectors_per_tick
soft_feedback_soft_remediate_rate
soft_feedback_critical_auto_rate
```

## Proof

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --json
```

Thunder locked. Yoi ⚡
