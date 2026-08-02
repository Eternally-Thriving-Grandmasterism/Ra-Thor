# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.15  
**License:** AG-SML v1.0  
**Contact:** info@Rathor.ai

## Stack

| Layer | Version | Role |
|-------|---------|------|
| Ambient → composite score | 0.5.0–0.5.12 | full algebra + health_score |
| Score gate + telemetry | 0.5.13 | demo CI ≥0.5 · Powrush health_score |
| ZoneHealthStatus | 0.5.14 | Healthy / Stressed / Critical |
| Dual-repo status mirror | Powrush v18.25 / orch v21.88.9 | H/S/C telemetry |
| Critical auto-remediate | 0.5.15 | priority Cosmic Tick on Critical zones |

## Critical auto-remediation

When `critical_auto_remediate = true` (default), any zone classified **Critical** after `process`/`ingest` is immediately purified. Counter: `critical_auto_purify_count` / `critical_auto_purifies`.

See [DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md](./DUAL_REPO_SOFT_FEEDBACK_CONTRACT.md).

29 property tests. Thunder locked. Yoi ⚡
