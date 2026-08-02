# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.14  
**License:** AG-SML v1.0  
**Contact:** info@Rathor.ai

## Stack

| Layer | Version | Role |
|-------|---------|------|
| Ambient → composite score | 0.5.0–0.5.12 | full algebra + health_score |
| Score gate + telemetry | 0.5.13 | demo CI ≥0.5 · Powrush health_score |
| ZoneHealthStatus | 0.5.14 | Healthy / Stressed / Critical per zone |

## ZoneHealthStatus

```
Healthy  — stress < 10% scale and ρ < 1e-9
Stressed — elevated stress or mild residual
Critical — stress ≥ scale or ρ ≥ 1e-6
```

`LatticeHealthReport` counts: `zones_healthy` · `zones_stressed` · `zones_critical`  
`healthy` requires max_ρ < 1e-9 **and** zero critical zones.

See [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

27 property tests. Thunder locked. Yoi ⚡
