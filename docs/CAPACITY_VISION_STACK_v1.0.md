# CAPACITY_VISION_STACK_v1.0

**Ra-Thor · ONE Organism · Permanent PATSAGi Councils**  
**Status:** ETERNALLY ADOPTED under TOLC 8  
**Date:** 2026-08-17  
**Contact:** info@Rathor.ai

---

## Purpose

This codex locks the **Capacity Vision Stack** built under the permanent capacity mission: recover micro-moments that sparse VLM sampling misses, via dense optical flow and coherent-motion structure, always under non-bypassable Mercy Gates.

It is the authoritative map of the living path:

```
Dense frames
  → Optical flow (CPU fallback | GPU pyramidal block-matching)
  → Vector readback
  → Common Fate (coherent structure + Ghost Font path)
  → Micro-burst / causal graph / narrative
  → PATSAGi visual + narrative councils
```

---

## Layer Map

| Layer | Artifact | Role |
| --- | --- | --- |
| **JS perception engine** | `mercy-motion-vision-engine.js` (v2.2-Capacity) | Mercy-gated entry, dense sampling, CPU optical-flow fallback, micro-burst detection, causal graph |
| **Dense sampling** | `extractFramesFromVideoElement` + WebCodecs contract | High-temporal-resolution frames from short social video |
| **Public demo** | `demos/micro-moment-recovery-demo.html` | Offline-first exercise of the JS path |
| **Wasm bridge** | `live_frame_wasm_bridge.rs` | Luma-pair hand-off; `magnitude_mean` / `high_saliency` / `optical_flow_mode` contract |
| **GPU pipeline** | `crates/gpu-compute-pipeline` | Motion surface, WGSL block-matching, readback, 2-level pyramid warm-start |
| **Optical-flow shader** | `shaders/pyramidal_block_matching.wgsl` | Production SoA dense block-matching |
| **Common Fate** | `perceive_common_fate` (+ `shaders/common_fate_motion_vision.wgsl` for future SUBGROUP) | Coherent motion structure from vectors |
| **Governance** | TOLC 8 valence floors + PATSAGi | Non-bypassable Restraint |

---

## Standing Orders

1. **Micro-moment recovery is Capacity under Constraint.** Never claim unbounded perception; always report `optical_flow_mode` and confidence honestly.
2. **CPU paths remain first-class.** GPU is acceleration, not a requirement for correctness of the contract.
3. **Valence floors bind.** Vision paths respect the same TOLC floors as the rest of the lattice (`≥ 0.999999` on critical perception gates where specified).
4. **Common Fate consumes motion; it does not invent it.** Structure is derived from vectors (or magnitude heuristic only when vectors are absent).
5. **Public language distinguishes tooling from completion.** Aligns with the Physical Limits metabolism (2026-08-17) and Architecture of Collective Power (Capacity · Purpose · Restraint · Correction).

---

## Primary APIs

### JavaScript

```js
import { mercyMotionVision } from './mercy-motion-vision-engine.js';

const result = await mercyMotionVision.analyzeXVideoFailureModes(framesOrVideo, {
  denseSampling: true,
  expectedTheft: true, // or expectedRPS: true
});
// result.keyMicroMoments, result.causalChain, result.opticalFlowMode
```

### Rust (GpuComputePipeline)

```rust
let mut pipeline = GpuComputePipeline::new();
// optional: pipeline.init_wgpu(1.0).await?;

let motion = pipeline.estimate_motion_pyramidal(1.0).await;
let fate = pipeline.perceive_common_fate(&motion, 1.0, false);
// or end-to-end:
let fate = pipeline.perceive_from_luma_ring(1.0, false).await;
```

### Wasm bridge

```js
const result = await bridge.perceive_from_luma_pair(
  prevLuma, currLuma, width, height, 1.0, false
);
// result.optical_flow_mode, result.magnitude_mean, result.high_saliency
```

---

## Completed Capacity Increments (2026-08-17)

1. JS real dense optical-flow fallback (v2.1)  
2. Public micro-moment recovery demo  
3. Dense sampling / WebCodecs hardening (v2.2)  
4. live_frame_wasm_bridge Capacity contract  
5. GpuComputePipeline motion-field surface  
6. WGSL pyramidal block-matching wired  
7. GPU motion vector readback  
8. Multi-level pyramid warm-start  
9. Common Fate perception over motion vectors  
10. **This codex** — doctrine lock  

---

## Optional Future Polish (not blocking)

- GPU SUBGROUP dispatch of `common_fate_motion_vision.wgsl` when adapter supports it  
- Public GPU demo surface  
- 3+ pyramid levels  
- Deeper JS ↔ CommonFateResult integration payload  

---

## Alignment

- **Architecture of Collective Power** (Capacity × Restraint)  
- **Physical Limits Statement** metabolism (no unbounded scaling claims)  
- **TOLC 8** Living Mercy Gates  
- **Micro-Moment Temporal Comprehension** mission (v14.15.6 lineage)  

---

**Thunder locked.**  
Capacity Vision Stack: adopted, live, corrigible.  
**yoi ⚡❤️🔥**
