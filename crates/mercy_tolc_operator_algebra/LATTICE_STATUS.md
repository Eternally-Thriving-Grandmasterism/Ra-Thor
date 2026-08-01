# mercy_tolc_operator_algebra — Lattice Status

**Version:** 0.5.9  
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
| Stress EMA recovery | 0.5.9 | Period recovers under calm (no permanent lock) |

## Adaptive Cosmic Tick (stress-driven)

```
stress_ema ← (1−α)·stress_ema + α·load
effective_period(z) = max(min_period, base / (1 + stress_ema / scale))
```

- `grief_absorbed` — cumulative telemetry (never decays)
- `stress_ema` — recent stress driving the Cosmic Tick (recovers under calm)
- Sibling zones receive mild stress decay each tick

Defaults: `purify_period=2500`, `adaptive_grief_scale=500`, `min_purify_period=50`, `stress_alpha=0.05`.

## Dual-repo

- **Ra-Thor:** `SoftFeedbackBridge` / `LatticeHealthReport`
- **Powrush-MMO:** `RaThorBridge::report_zone_grief` + orchestrator tick + telemetry

Sealed event fields: `{ zone_id, grief_load, valence, under_floor, tick }`

## Public proofs

```bash
cargo test -p mercy_tolc_operator_algebra
cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --json
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench -- --agents 50000 --zones 5
```

23 property tests. Thunder locked. Yoi ⚡
