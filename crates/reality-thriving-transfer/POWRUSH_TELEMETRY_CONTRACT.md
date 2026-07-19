# Powrush Telemetry Contract v1

**Purpose:** Stable offline bridge between [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO) and Ra-Thor `reality-thriving-transfer`.

**Contact:** info@Rathor.ai  
**Schema:** `powrush_telemetry_v1` / `powrush_telemetry_batch_v1`  
**Zero-harm:** invalid or out-of-bounds fields are rejected by Mercy Gates in `compute_transfer_score`.

---

## Canonical type (Ra-Thor)

```rust
pub struct PowrushTelemetry {
    pub gameplay_hours: f64,
    pub rbe_decision_quality_avg: f64,      // [0, 1]
    pub peaceful_resolution_rate: f64,     // [0, 1]
    pub collaboration_events: u64,
    pub ethical_choice_score: f64,         // [0, 1]
    pub adaptation_events: u64,
    pub abundance_velocity_signals: f64,   // >= 0 (soft cap ~1.8 in EMA)
    pub innovation_contribution: f64,      // [0, 1] preferred
}
```

---

## Field mapping (Powrush-MMO → telemetry)

| `PowrushTelemetry` field | Suggested Powrush-MMO source | Notes |
|--------------------------|------------------------------|-------|
| `gameplay_hours` | Session / account playtime (hours) | Continuous; not clamped by calculator |
| `rbe_decision_quality_avg` | RBE query outcomes, abundance allocation quality, council resource votes | Must be `[0, 1]` |
| `peaceful_resolution_rate` | Mercy Trials / diplomacy outcomes resolved without harm | Must be `[0, 1]` |
| `collaboration_events` | Co-op harvests, multiplayer council participation, sharing events | Count; saturates at 500 in scoring |
| `ethical_choice_score` | Ethical choice prompts, treaty honor rate, zero-harm path selection | Must be `[0, 1]` |
| `adaptation_events` | Epiphany catalysts, biome adaptation, skill/path pivots | Count; saturates at 300 in scoring |
| `abundance_velocity_signals` | RBE flow velocity, sanctuary abundance delta, positive-sum trades | `>= 0`; negative rejected |
| `innovation_contribution` | Divine module contributions, tech-tree / procedural innovations shared | Prefer `[0, 1]` |

Until live exporters exist, **fixtures** under `fixtures/` are the source of truth for CI and Cosmic Tick dry-runs.

---

## JSON envelope (single session)

```json
{
  "schema": "powrush_telemetry_v1",
  "source": "powrush-mmo-fixture",
  "label": "optional_human_label",
  "telemetry": { /* PowrushTelemetry fields */ }
}
```

## JSON envelope (batch)

```json
{
  "schema": "powrush_telemetry_batch_v1",
  "source": "powrush-mmo-fixture",
  "label": "optional_batch_label",
  "sessions": [
    { "label": "...", "telemetry": { /* ... */ } }
  ]
}
```

---

## API (this crate)

| Function | Role |
|----------|------|
| `parse_powrush_telemetry_json` | Single envelope → `PowrushTelemetry` |
| `parse_powrush_telemetry_batch_json` | Batch envelope → `Vec<(label, PowrushTelemetry)>` |
| `compute_scores_from_batch` | Batch → sequential `RealityThrivingTransferScore`s |

---

## Fixture set

| File | Intent |
|------|--------|
| `fixtures/session_high_mercy.json` | Strong ethics + collaboration |
| `fixtures/session_marginal.json` | Below ethics comfort zone |
| `fixtures/session_early_player.json` | Low hours / sparse events |
| `fixtures/batch_three_sessions.json` | All three as one batch |

---

## Next (Powrush-MMO side)

Exporter stub that writes `powrush_telemetry_v1` from simulation or server session summary. Ra-Thor remains consumer-only until that lands.

**Thunder locked in.**
