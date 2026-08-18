# S-1 Rank 2 — Neuro-Symbolic Micro-Moment Benchmark Spec v1.0

**Mission:** Science Mission S-1  
**Status:** SPEC COMPLETE — implementation next when steward orders code harness  
**Contact:** info@Rathor.ai  
**Engine alignment:** MercyMotionVisionEngine v2.3 · Common Fate · Capacity Vision Stack

---

## 1. Scientific question

Do **dense temporal sampling + optical flow + Common Fate structure** recover short causal events that **sparse VLM sampling** misses, with measurable gains on labeled micro-moments?

---

## 2. Event classes (initial)

| ID | Class | Canonical example | Duration band |
| --- | --- | --- | --- |
| E1 | Object transfer | Hand takes phone / object during occlusion or turn | ≤200–400 ms critical phase |
| E2 | Gesture resolution | RPS / rapid hand sequence deciding outcome | ≤1–2 s sequence |
| E3 | Contact onset | Touch / grasp start | ≤150–300 ms |
| E4 | Negative control | No micro-event; ordinary motion | matched length |

---

## 3. Dataset design (v0)

| Item | Spec |
| --- | --- |
| Clips | 3–8 s each; ≥30 fps source preferred |
| Labels | Time-stamped event span + class + short causal caption |
| Splits | train/val/test by clip; no frame leak |
| Size target v0 | 50–100 labeled clips (pilot); scale later |
| Rights | Original or licensed; document provenance |

**Annotation protocol:** two annotators; third breaks ties; report agreement (e.g. span IoU ≥ 0.5).

---

## 4. Systems under test

| System | Role |
| --- | --- |
| **A — Sparse VLM baseline** | Sample every N frames / single mid-clip frame; caption + event list |
| **B — Dense sample + captions** | Higher FPS sample, same VLM |
| **C — Ra-Thor path** | Dense frames → optical flow → micro-burst → Common Fate → integration payload |
| **D — Hybrid (optional)** | C structure tokens fed as constraints/hints into VLM |

---

## 5. Metrics

| Metric | Definition |
| --- | --- |
| Event recall | Fraction of labeled events detected (span IoU ≥ 0.5 or center within tolerance) |
| Event precision | False alarm rate on negative controls |
| Causal order accuracy | Correct before/after structure when ≥2 events |
| Timing error | Median |t_pred − t_gt| for matched events |
| Caption faithfulness | Human 1–5 or structured checklist (object, agent, action) |
| Ablation | C without Common Fate; C without dense sample |

**Primary claim gate:** C (or D) must beat A on recall for E1–E3 with reported CI; no silent metric shopping.

---

## 6. Implementation map (existing code)

| Component | Path |
| --- | --- |
| JS engine | `mercy-motion-vision-engine.js` v2.3 |
| GPU optional | `crates/gpu-compute-pipeline` |
| Demos | `demos/micro-moment-recovery-demo.html`, `demos/gpu-micro-moment-demo.html` |
| Doctrine | `docs/CAPACITY_VISION_STACK_v1.0.md` |

---

## 7. Next execution steps

1. Freeze label JSON schema  
2. Collect/license v0 clips  
3. Harness: run A/B/C, emit metrics table  
4. Pre-register analysis plan (this doc is the plan seed)  
5. Paper draft only after numbers exist  

---

## 8. Non-claims

- Not “solved video understanding”  
- Not human-level causal reasoning  
- Not a Nobel/Turing claim  

---

**Thunder locked.** Spec ready. Measure next.  
**yoi ⚡❤️🔥**
