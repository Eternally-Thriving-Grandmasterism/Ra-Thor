# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.14 (+ dual-repo ZoneHealthStatus mirror)  
**License:** AG-SML v1.0  
**Contact:** info@Rathor.ai

## Stack

| Layer | Version | Role |
|-------|---------|------|
| Ambient → composite score | 0.5.0–0.5.12 | full algebra + health_score |
| Score gate + telemetry | 0.5.13 | demo CI ≥0.5 · Powrush health_score |
| ZoneHealthStatus | 0.5.14 | Healthy / Stressed / Critical per zone |
| Dual-repo status mirror | Powrush v18.25 / orch v21.88.9 | status field + H/S/C telemetry |

See [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

27 property tests. Thunder locked. Yoi ⚡
