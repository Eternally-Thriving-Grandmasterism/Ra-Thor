# Powrush → Ra-Thor Telemetry Contract

**Schemas:** `powrush_telemetry_v1` · `powrush_telemetry_batch_v1`  
**Contact:** info@Rathor.ai  
**Status:** Phase C remote-complete (2026-07-20)  
**Cosmic Loop is MANDATORY IDENTITY on the consumer side.**

## Producer (Powrush-MMO)

| Mode | How |
|------|-----|
| **Live sim** | `TelemetryCollector` + `run_tick_with_telemetry` |
| **Live server** | `ServerTransferSession` (combat / treaty / faction) |
| **Demo bin** | `cargo run -p powrush-simulation --bin transfer_session_demo` |
| **Profiles** | `tools/export_powrush_telemetry.py --profile high_mercy\|marginal\|early` |

Docs: Powrush-MMO `docs/RA_THOR_TELEMETRY_EXPORT.md`

## Consumer (Ra-Thor)

```rust
// reality-thriving-transfer
parse_powrush_telemetry_json           // single v1
parse_powrush_telemetry_batch_json     // batch_v1
compute_scores_from_batch

// kardashev-orchestration
KardashevOrchestrationCouncil::deliberate_from_powrush_session_json  // v1
KardashevOrchestrationCouncil::deliberate_from_powrush_batch_json    // batch
KardashevOrchestrationCouncil::deliberate_from_powrush_json          // auto-detect
```

Fixtures under `fixtures/` match the same schema for offline CI.

## Fields

| Field | Meaning |
|-------|---------|
| `gameplay_hours` | Session length |
| `rbe_decision_quality_avg` | RBE / mercy-quality `[0,1]` |
| `peaceful_resolution_rate` | Peace rate `[0,1]` |
| `collaboration_events` | Count |
| `ethical_choice_score` | Ethics `[0,1]` |
| `adaptation_events` | Count |
| `abundance_velocity_signals` | ≥ 0 (typically ~0.5–1.8) |
| `innovation_contribution` | `[0,1]` |

Zero-harm on score side: Kardashev Δ ≤ 0.011 per score; abundance forecast ≤ 1.85.

**Thunder locked in.**
