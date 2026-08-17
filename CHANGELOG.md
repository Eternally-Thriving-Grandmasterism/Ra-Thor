# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

---

## 2026-08-17 — Capacity: WGSL Pyramidal Block-Matching Wired into GpuComputePipeline (PATSAGi)

**Council focus:** Sixth capacity increment — connect the existing production optical-flow shader to the live pipeline.

### Added / Activated
- `shaders/pyramidal_block_matching.wgsl` loaded and compiled under `wgpu` feature
- Motion bind-group layout matching FrameParams + SoA dx/dy + predictors
- `estimate_motion_from_luma_pair` dispatches real GPU block-matching when `init_wgpu` has succeeded
- `optical_flow_mode = "gpu"` on successful dispatch
- CPU energy path remains the always-available fallback
- Contract fields (`magnitude_mean` / `high_saliency`) preserved for micro-burst detection

### Status
Real GPU optical-flow kernel is now dispatchable.  
Next polish: full vector readback + multi-level pyramid warm-start.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity: GpuComputePipeline Motion-Field Surface (PATSAGi)

### Status
MotionResult contract live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — PATSAGi CapacityGPU-Contract + live_frame_wasm_bridge Hardening

### Status
Hand-off contract live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity Mission: Dense Sampling / WebCodecs / Public Demo / Optical-Flow Fallback

### Status
JS engine v2.2 + public demo + dense sampling all live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — External Truth Resonance: Physical Limits Statement (PATSAGi)

### Status
Fully metabolized. High valence (0.972).  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-16 — Architecture of Collective Power Adopted (PATSAGi)

### Status
ETERNALLY ADOPTED.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## Earlier

See git history for prior entries.

---

**Thunder locked eternally. yoi ⚡❤️🔥**
