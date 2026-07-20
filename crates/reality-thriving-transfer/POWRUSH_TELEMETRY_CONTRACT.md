# Powrush → Ra-Thor Telemetry Contract

**Schemas:** `powrush_telemetry_v1` · `powrush_telemetry_batch_v1`  
**Contact:** info@Rathor.ai  
**Cosmic Loop is MANDATORY IDENTITY on the consumer side.**

## Producer (Powrush-MMO)

| Mode | How |
|------|-----|
| **Live counters** | `TelemetryCollector` embeds `GlobalTransferSession`; `collect_tick` / `record_tick_result` accumulate every sim tick |
| **Demo bin** | `cargo run -p powrush-simulation --bin transfer_session_demo` |
| **Profiles** | `tools/export_powrush_telemetry.py --profile high_mercy\|marginal\|early` |

Docs: Powrush-MMO `docs/RA_THOR_TELEMETRY_EXPORT.md`

## Consumer (Ra-Thor)

```rust
// reality-thriving-transfer
parse_powrush_telemetry_json / parse_powrush_telemetry_batch_json
compute_scores_from_batch

// kardashev-orchestration
KardashevOrchestrationCouncil::deliberate_from_powrush_batch_json(json, None)
```

Fixtures under `fixtures/` match the same schema for offline CI.

## Fields

| Field | Meaning |
|-------|---------|
| `gameplay_hours` | Session length |
| `rbe_decision_quality_avg` | RBE / mercy-quality of decisions `[0,1]` |
| `peaceful_resolution_rate` | Conflict resolution peace rate `[0,1]` |
| `collaboration_events` | Count |
| `ethical_choice_score` | Ethics `[0,1]` |
| `adaptation_events` | Count |
| `abundance_velocity_signals` | Abundance pressure (unbounded ≥ 0; typically ~0.5–1.8) |
| `innovation_contribution` | Innovation `[0,1]` |

Zero-harm on score side: Kardashev Δ ≤ 0.011 per score; abundance forecast ≤ 1.85.

**Thunder locked in.**
