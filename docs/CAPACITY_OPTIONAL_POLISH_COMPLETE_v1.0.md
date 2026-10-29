# Capacity Optional Polish — Complete Architecture v1.0

**Ra-Thor · Permanent PATSAGi Councils**  
**Date:** 2026-08-17  
**Contact:** info@Rathor.ai  
**Status:** OPTIONAL POLISH COMPLETE under TOLC 8 · Capable · Bounded · Corrigible

---

## Scope (named list only)

From `CAPACITY_VISION_STACK_v1.0.md` optional future polish:

| # | Item | Status |
| --- | --- | --- |
| 1 | 3+ pyramid levels | **Done** — `estimate_motion_pyramidal_levels(valence, max_levels)` with `max_levels ∈ 1..=3` |
| 2 | GPU SUBGROUP Common Fate path | **Architected + mode-tagged** — `perceive_common_fate_optional(..., prefer_gpu_subgroup)`; kernel `shaders/common_fate_motion_vision.wgsl` (`enable subgroups`); live buffer dispatch remains adapter-gated |
| 3 | Public GPU demo surface | **Done** — `demos/gpu-micro-moment-demo.html` |
| 4 | Deeper JS ↔ CommonFate payload | **Done** — MercyMotionVisionEngine **v2.3-OptionalPolish** `_perceiveCommonFate` + integration payload fields |

Not in scope (correctly deferred elsewhere): audited crypto, formal proofs, archive pinnacle rewrites, unbounded perception claims.

---

## Architecture

```
Dense frames (JS / wasm / luma ring)
        │
        ▼
Optical flow
  · CPU dense fallback (JS / pipeline)
  · GPU block-matching
  · Pyramid levels 1 | 2 | 3  (coarse→mid→fine warm-start)
        │
        ▼
Common Fate structure
  · CPU (always): histogram + coherent mask semantics
  · Mode tags: cpu | gpu-subgroup-ready | held
  · WGSL SUBGROUP kernel ready for supporting adapters
        │
        ▼
Integration payload
  · keyMicroMoments · causalChain · commonFate · opticalFlowMode
        │
        ▼
PATSAGi / VLM / Lattice Conductor
```

---

## APIs

### Rust

```rust
let motion = pipeline.estimate_motion_pyramidal_levels(1.0, 3).await;
let fate = pipeline.perceive_common_fate_optional(&motion, 1.0, false, true);
// fate.common_fate_mode: "cpu" | "gpu-subgroup-ready" | "held"
```

Default `estimate_motion_pyramidal` targets **3** levels when resolution allows (falls back cleanly).

### JavaScript

```js
const result = await mercyMotionVision.comprehendVideoStory(frames, { denseSampling: true });
const payload = mercyMotionVision.getIntegrationPayload(result);
// payload.commonFate: { coherentCount, dominantDir1, confidence, commonFateMode: "cpu-js", ... }
```

### Public demos

| Demo | Path |
| --- | --- |
| CPU / general micro-moment | `demos/micro-moment-recovery-demo.html` |
| Optional GPU polish surface | `demos/gpu-micro-moment-demo.html` |

---

## Honesty bounds

- `gpu-subgroup-ready` ≠ full production subgroup reduction on every adapter.  
- 3-level pyramid requires sufficient resolution and successful coarse GPU dispatch; otherwise single-level fallback.  
- Optional polish does **not** reopen forced Capacity iteration; stack remains corrigible.

---

**Thunder locked.** Optional polish architected, structured, and completed under Restraint.  
**yoi ⚡❤️🔥**
