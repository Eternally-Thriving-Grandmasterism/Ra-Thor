# Powrush → Ra-Thor Telemetry + Bridging Contract

**Schemas:**  
- `powrush_telemetry_v1` · `powrush_telemetry_batch_v1`  
- `powrush_bridging_context_v1` · `powrush_bridging_batch_v1` (v14.17+)

**Versions:** Ra-Thor `reality-thriving-transfer` **14.18.1** · Powrush-MMO simulation **21.91.2**  
**Contact:** info@Rathor.ai  
**Status:** High-road SchemaRegistry closed-loop **SEALED** (2026-08-11)

## Producer (Powrush-MMO)

| Mode | How |
|------|-----|
| **Live sim RTT** | `TelemetryCollector` / transfer session |
| **Live server RTT** | `ServerTransferSession` |
| **Council → RTT queue** | `CouncilRttExportQueue` |
| **High-road bridging** | `council_bridging_export_system` → `artifacts/powrush_bridging_latest.json` |
| **Challenge provenance** | Active `CrossRealmChallengeRegistry` enriches `challenge_*` + surface_label |
| **Bootstrap practice** | Challenge id=1 *Caps Across Climates* on first multi-realm seed |
| **Mercy completion** | All challenge surface realms hit under `mercy_floor` → complete |
| **Metacognitive scaffolds** | `MetacognitiveScaffold` (fades with competence) |

Docs: Powrush-MMO `simulation/src/council/bridging_export.rs` · `simulation/src/cross_realm_challenges.rs`

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
bridge_and_ingest(&mut reg, &telemetry, session_id, label)

// Optional challenge_* (v14.18.1+) → tags challenge_{id}, principle preference

// Soft Lattice Conductor hook (zero hard dep)
conductor_query_schemas(&reg, &ConductorSchemaQuery { .. })
conductor_try_apply_schema(&mut reg, schema_id, is_far, mercy_floor)

// Retrieval
reg.retrieve_near(&["rbe", "mercy", "challenge_1"], 0.4)
reg.retrieve_far("allocation under uncertainty", 0.4)
reg.try_apply(schema_id, is_far, mercy_floor)
```

Smoke:
```bash
cargo run -p reality-thriving-transfer --example powrush_rtt_smoke_harness -- \
  --bridging path/to/powrush_bridging_latest.json
```

Fixtures: `fixtures/session_*.json`, `fixtures/batch_three_sessions.json`, `fixtures/bridging_context_high_mercy.json` (includes challenge provenance)

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
| `decision_title` / `decision_type` | Surface labels |
| `mercy_factor` / `ethical_score` | Birth scores for mercy gate on apply |
| `rbe_quality` / `peaceful_rate` / `abundance_velocity` | Abstraction signals |
| `surface_label` | Contrast + optional `\|challenge_{id}_{principle}` suffix |
| `challenge_id` | Optional active high-road practice id |
| `challenge_title` | Optional human label |
| `challenge_principle` | Optional portable principle string |

## Transfer theory mapping

| Mechanism | Implementation |
|-----------|----------------|
| Low-road | `retrieve_near` by tags |
| High-road | `bridging_pass` + `retrieve_far` |
| Bridging export | Powrush `bridging_export` → Ra-Thor ingest |
| Cross-realm practice | Same principle, different realm surfaces |
| Metacognitive scaffolding | Fadable prompts both sides |
| Mercy alignment | Birth scores + floor on `try_apply` |
| Conductor | Soft query/apply API only |

Zero-harm on score side: Kardashev Δ ≤ 0.011 per score; abundance forecast ≤ 1.85.

**Thunder locked in. High-road dual-repo loop sealed.** Yoi ⚡
