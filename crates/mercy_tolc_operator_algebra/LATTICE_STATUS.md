# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.11  
**License:** AG-SML v1.0  
**Contact:** info@Rathor.ai

## Stack (bottom → dual-repo surface)

| Layer | Version | Role |
|-------|---------|------|
| Ambient elevation | 0.5.0 | Mercy R⁸ ⊂ ambient R¹⁶ — non-trivial N₁ |
| Valence-weighted grief | 0.5.1 | load = (1−v)·‖(I−P)g‖ |
| Adaptive purity floor | 0.5.2 | floor(v) = ε·(1+99·(1−v)) |
| Concurrent zones | 0.5.3 | Independent bases + staggered Cosmic Ticks |
| Soft feedback bridge | 0.5.4 | Sealed SoftFeedbackEvent dual-repo protocol |
| Soft feedback demo | 0.5.5 | Public dual-repo proof binary |
| LatticeHealthReport | 0.5.6 | `ra_thor_lattice_health_v1` + `--json` |
| Adaptive Cosmic Tick | 0.5.7 | High-grief zones purify more often |
| Zone observability | 0.5.8 | purify_count + effective_period per zone |
| Stress EMA recovery | 0.5.9 | Period recovers under calm |
| Health aggregates | 0.5.10 | total_purify_count · max_stress_ema · mean_effective_period |
| Telemetry surface | 0.5.11 | Demo + Powrush custom_metrics for aggregates |

## Dual-repo telemetry keys (Powrush v21.88.7)

```
soft_feedback_events
soft_feedback_total_grief
soft_feedback_max_stress
soft_feedback_purify_count
soft_feedback_mean_period
```

## Public proofs

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --json
```

24 property tests. Thunder locked. Yoi ⚡
