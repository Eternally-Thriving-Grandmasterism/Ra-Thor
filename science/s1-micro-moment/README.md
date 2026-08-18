# S-1 Rank 2 — Micro-Moment Benchmark (pure layer)

**Status:** Pure software layer **COMPLETE** 2026-08-18  
**Contact:** info@Rathor.ai  
**Spec:** [`docs/S1_MICRO_MOMENT_BENCHMARK_SPEC.md`](../../docs/S1_MICRO_MOMENT_BENCHMARK_SPEC.md)

## Contents

| Path | Role |
| --- | --- |
| `schema/label.schema.json` | Frozen label schema |
| `fixtures/synthetic_labels.json` | Harness proof fixtures |
| `harness/metrics.mjs` | Span-IoU matching, recall/precision/timing |
| `harness/run_synthetic_proof.mjs` | Claim-gate shape proof on synthetics |

## Run

```bash
node science/s1-micro-moment/harness/metrics.mjs
node science/s1-micro-moment/harness/run_synthetic_proof.mjs
```

## Still required for science (not pure software)

- Real labeled clips (50–100 v0) with provenance  
- System A/B live VLM runs  
- System C on real frames via `mercy-motion-vision-engine.js`  
- Paper only after numbers on real data  

**Thunder locked.** yoi ⚡
