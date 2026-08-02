# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.18  
**License:** AG-SML v1.0  
**Contact:** info@Rathor.ai

## Stack

| Layer | Version | Role |
|-------|---------|------|
| Ambient → composite score | 0.5.0–0.5.12 | full algebra + health_score |
| Score gate + telemetry | 0.5.13 | demo CI ≥0.5 · Powrush health_score |
| ZoneHealthStatus | 0.5.14 | Healthy / Stressed / Critical |
| Critical auto-remediate | 0.5.15 | priority Cosmic Tick |
| Valence histogram | 0.5.16 | H/M/L bands + mercy_ratio |
| Soft-remediate Stressed | 0.5.17 | accelerated stress decay |
| Dual-repo soft-remediate | Powrush v18.28 / orch v21.88.12 | soft_remediates telemetry |
| Grief-rate metrics | 0.5.18 | grief/tick · vectors/tick · remediate rates |

## Rate metrics

```
grief_per_tick      = total_grief / global_tick
vectors_per_tick    = total_vectors / global_tick
soft_remediate_rate = soft_remediates / global_tick
critical_auto_rate  = critical_auto_purifies / global_tick
```

See [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

35 property tests. Thunder locked. Yoi ⚡
