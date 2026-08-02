# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.16 (+ dual-repo valence histogram mirror)  
**License:** AG-SML v1.0  
**Contact:** info@Rathor.ai

## Stack

| Layer | Version | Role |
|-------|---------|------|
| Ambient → composite score | 0.5.0–0.5.12 | full algebra + health_score |
| Score gate + telemetry | 0.5.13 | demo CI ≥0.5 · Powrush health_score |
| ZoneHealthStatus | 0.5.14 | Healthy / Stressed / Critical |
| Critical auto-remediate | 0.5.15 | priority Cosmic Tick |
| Dual-repo critical mirror | Powrush v18.26 / orch v21.88.10 | critical_auto telemetry |
| Valence histogram | 0.5.16 | H/M/L bands + mercy_ratio |
| Dual-repo valence mirror | Powrush v18.27 / orch v21.88.11 | valence + mercy_ratio telemetry |

See [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

31 property tests. Thunder locked. Yoi ⚡
