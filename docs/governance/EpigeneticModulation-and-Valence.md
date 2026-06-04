# EpigeneticModulation and Valence Effects

**Version:** v14.6

## Purpose

`EpigeneticModulation` is the mechanism by which real PATSAGi Council valence directly influences evolutionary parameters (strength, volatility, evolution rate) in the geometric layer.

## Key Methods

- `apply_council_valence(valence, council)`
- `explore_valence_impact(valence, council)` — rich before/after report
- `simulate_council_sequence(sequence)` — cumulative effects
- `visualize_valence_history_ascii()`
- `export_valence_history_json()`

## Valence Effects Summary

| Council Type       | Primary Effect          | Secondary Effect       |
|--------------------|-------------------------|------------------------|
| Evolutionary       | Strong strength boost   | High evolution rate    |
| Infinite           | Strong strength boost   | Sustained growth       |
| Harmony / Truth    | Volatility reduction    | Long-term stabilization|
| Others             | Moderate strength boost | Balanced influence     |

Higher geometric layers (especially Hyperbolic) amplify positive valence effects.

## Integration

Used by:
- `RiemannianMercyManifold`
- `ShardManager` per-shard state
- Simulation ticks and proposal evaluation

See also: `PATSAGi-Council-Engine.md` and `ShardManager-and-Interest-Management.md`.
