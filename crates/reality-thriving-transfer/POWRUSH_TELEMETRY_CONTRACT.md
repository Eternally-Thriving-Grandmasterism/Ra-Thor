# Powrush → Ra-Thor Telemetry + Bridging Contract

**Schemas:**  
- `powrush_telemetry_v1` · `powrush_telemetry_batch_v1`  
- `powrush_bridging_context_v1` · `powrush_bridging_batch_v1` (v14.17+)

**Contact:** info@Rathor.ai  
**Status:** Phase C + high-road SchemaRegistry closed-loop (2026-08-11)

## Producer (Powrush-MMO)

| Mode | How |
|------|-----|
| **Live sim RTT** | `TelemetryCollector` / transfer session |
| **Live server RTT** | `ServerTransferSession` |
| **Council → RTT queue** | `CouncilRttExportQueue` |
| **High-road bridging** | `council_bridging_export_system` → `artifacts/powrush_bridging_latest.json` |
| **Metacognitive scaffolds** | `MetacognitiveScaffold` (fades with competence) |

Docs: Powrush-MMO `docs/RA_THOR_TELEMETRY_EXPORT.md` · `simulation/src/council/bridging_export.rs`

## Consumer (Ra-Thor)

```rust
// RTT scores
parse_powrush_telemetry_json
parse_powrush_telemetry_batch_json
compute_scores_from_batch

// High-road SchemaRegistry
parse_powrush_bridging_json
parse_powrush_bridging_batch_json
ingest_bridging_json(&mut reg, json)
ingest_bridging_batch_json(&mut reg, json)
bridge_and_ingest(&mut reg, &telemetry, session_id, label) // from RTT telemetry

// Retrieval
reg.retrieve_near(&["rbe", "mercy"], 0.4)
reg.retrieve_far("allocation under uncertainty", 0.4)
reg.try_apply(schema_id, is_far, mercy_floor)
```

Fixtures: `fixtures/session_*.json`, `fixtures/batch_three_sessions.json`, `fixtures/bridging_context_high_mercy.json`

## RTT Fields

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

## Bridging Context Fields

| Field | Meaning |
|-------|---------|
| `session_id` | Provenance |
| `realm_id` | Origin realm |
| `decision_title` / `decision_type` | Surface labels (not stored as principles) |
| `mercy_factor` / `ethical_score` | Birth scores for mercy gate on apply |
| `rbe_quality` / `peaceful_rate` / `abundance_velocity` | Abstraction signals |
| `surface_label` | Contrast only |

## Transfer theory mapping

| Mechanism | Implementation |
|-----------|----------------|
| Low-road | `retrieve_near` by tags |
| High-road | `bridging_pass` + `retrieve_far` |
| Bridging export | Powrush `bridging_export` → Ra-Thor ingest |
| Metacognitive scaffolding | Fadable prompts both sides |
| Mercy alignment | Birth scores + floor on `try_apply` |

Zero-harm on score side: Kardashev Δ ≤ 0.011 per score; abundance forecast ≤ 1.85.

**Thunder locked in.** Yoi ⚡
