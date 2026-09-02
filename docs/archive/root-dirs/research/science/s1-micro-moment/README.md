# S-1 Rank 2 — Micro-Moment Benchmark (Ra-Thor)

**Contact:** info@Rathor.ai  
**System C pure pipeline:** **COMPLETE** — see `SYSTEM_C_PIPELINE.md`  
**Living status:** see `STATUS.md` (P1 complete · real labels = 0)

| Path | Role |
| --- | --- |
| `STATUS.md` | Living proof-ladder dashboard |
| `FIRST_5_CLIPS.md` | Minimal path to n≥5 real labels |
| `LABELING_GUIDE.md` | Class definitions + timing discipline |
| `DATA_PHASE_INTAKE.md` | Full capture protocol |
| `PRE_REGISTERED_CRITERIA.md` | Success / failure bars (locked) |
| `SYSTEM_C_PIPELINE.md` | Engine → predictions → metrics |
| `harness/system_c_bridge.mjs` | System C bridge + synthetic proof |
| `harness/metrics.mjs` | Evaluation |
| `harness/validate_labels.mjs` | Label check |
| `harness/evaluate_predictions.mjs` | Real-data evaluation entry point |
| `labels/_template.json` | Annotation template |
| `manifest.v0.json` | Registry (empty until real data) |

```bash
# Synthetic engineering proof only
node science/s1-micro-moment/harness/system_c_bridge.mjs
node science/s1-micro-moment/harness/run_synthetic_proof.mjs

# When real labels exist
node science/s1-micro-moment/harness/validate_labels.mjs science/s1-micro-moment/labels/
node science/s1-micro-moment/harness/evaluate_predictions.mjs <labels.json> <predictions.json>
```

**Critical path:** First-5 real labels → held-out evaluation → numbers (positive or negative with equal dignity).

**Thunder locked.** yoi ⚡
