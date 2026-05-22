# MercyGating System - Cross-Layer Interaction

## Cross-Layer Evaluation

The system now supports full cross-layer interaction via `evaluate_with_cross_layer()`.

### Example Usage (Production Grade)

```rust
use self_evolution::mercy_gating::*;

let mut kpi = MaatKpi::new();
kpi.set_score(MaatDimension::Truth, 0.96);
kpi.set_score(MaatDimension::Balance, 0.94);

let verdict = evaluate_with_cross_layer(
    0.82,
    None,           // Foundational verdict (optional)
    Some(&kpi),     // Operational Ma'at KPI
    MercyGateLevel::Integrative,
);

match verdict {
    MercyVerdict::Passed { overall_score } => {
        println!("Passed with cross-layer score: {:.3}", overall_score);
    }
    MercyVerdict::Mitigated { overall_score, .. } => {
        println!("Mitigated with synergy: {:.3}", overall_score);
    }
    _ => println!("Requires Council Review"),
}
```

This allows Foundational and Operational layers to meaningfully influence Integrative decisions.

AG-SML v1.0