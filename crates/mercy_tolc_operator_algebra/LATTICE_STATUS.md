# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.19 (Tikhonov-damped projector)  
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
| Dual-repo soft-remediate | Powrush v18.28 / orch v21.88.13 | soft_remediates telemetry |
| Grief-rate metrics | 0.5.18 | grief/tick · vectors/tick · rates |
| Dual-repo rate metrics | Powrush v18.29 / orch v21.88.13 | rate telemetry |
| Tikhonov projector | 0.5.19 | P_λ = E(EᵀE+λI)⁻¹Eᵀ · λ from ρ / stress · purify resets |

See [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

40+ property tests. Thunder locked. Yoi ⚡
