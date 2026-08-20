# S-1 Rank 2 — Micro-Moment Benchmark

**Contact:** info@Rathor.ai  
**Door 1:** Data phase intake **OPEN** — `DATA_PHASE_INTAKE.md`

| Path | Role |
| --- | --- |
| `DATA_PHASE_INTAKE.md` | Capture & label |
| `schema/label.schema.json` | Schema |
| `labels/_template.json` | Per-clip template |
| `manifest.v0.json` | Registry (empty = honest) |
| `harness/metrics.mjs` | Evaluation |
| `harness/validate_labels.mjs` | Structural check |
| `harness/run_synthetic_proof.mjs` | Smoke only |

```bash
node science/s1-micro-moment/harness/validate_labels.mjs science/s1-micro-moment/fixtures/synthetic_labels.json
node science/s1-micro-moment/harness/run_synthetic_proof.mjs
```

**Thunder locked.** yoi ⚡
