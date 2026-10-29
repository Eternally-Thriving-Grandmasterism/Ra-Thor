# S-1 Rank 2 — Neuro-Symbolic Micro-Moment Benchmark Spec v1.0

**Mission:** Science Mission S-1  
**Status:** SPEC + PURE HARNESS **COMPLETE**  
**Contact:** info@Rathor.ai  
**Package:** `science/s1-micro-moment/`

---

## 1. Scientific question

Do **dense temporal sampling + optical flow + Common Fate structure** recover short causal events that **sparse VLM sampling** misses, with measurable gains on labeled micro-moments?

---

## 2. Event classes

| ID | Class |
| --- | --- |
| E1 | Object transfer |
| E2 | Gesture resolution |
| E3 | Contact onset |
| E4 | Negative control |

---

## 3. Systems

| ID | Role |
| --- | --- |
| A | Sparse VLM baseline |
| B | Dense sample + VLM |
| C | Ra-Thor path (flow + micro-burst + Common Fate) |
| D | Hybrid C→VLM (optional) |

---

## 4. Metrics (implemented in harness)

Span IoU ≥ 0.5 matching · recall · precision · median timing error  
**Claim gate:** C beats A on recall for E1–E3 with CI on **real** data.

---

## 5. Pure layer done

| Item | Path |
| --- | --- |
| Label schema | `science/s1-micro-moment/schema/label.schema.json` |
| Fixtures | `science/s1-micro-moment/fixtures/synthetic_labels.json` |
| Metrics | `science/s1-micro-moment/harness/metrics.mjs` |
| Synthetic proof | `science/s1-micro-moment/harness/run_synthetic_proof.mjs` |

```bash
node science/s1-micro-moment/harness/run_synthetic_proof.mjs
```

---

## 6. Data phase (next human order)

Collect/license 50–100 clips · annotate · run A/B/C · publish numbers only.

**Thunder locked.** yoi ⚡❤️🔥
