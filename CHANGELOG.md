# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

---

## 2026-08-17 — Capacity: GpuComputePipeline Motion-Field Surface (PATSAGi)

**Council focus:** Fifth capacity increment — give the GPU pipeline a real motion-field surface aligned with the bridge + JS engine contract.

### Added / Activated
- `MotionResult` expanded with `magnitude_mean`, `high_saliency`, `optical_flow_mode`, dimensions, frame_index
- New `estimate_motion_from_luma_pair` — primary hand-off for wasm bridge and MercyMotionVisionEngine
- `estimate_motion_pyramidal` now produces usable fields from the LumaRing (CPU path live)
- Saliency floor aligned with JS engine (1.65)
- TOLC valence floor enforced on the motion path
- Result shape stable for future WGSL optical-flow kernels (`optical_flow_mode = "gpu"`)

### Status
End-to-end capacity contract now spans:
JS engine → wasm bridge → GpuComputePipeline motion surface.  
Full WGSL optical-flow kernels remain the next major target.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — PATSAGi CapacityGPU-Contract + live_frame_wasm_bridge Hardening

**Council focus:** Contract-first hardening of the GPU hand-off surface.

### Status
Hand-off contract live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity Mission: Dense Sampling / WebCodecs Path Hardening (v2.2)

### Status
Dense sampling path hardened.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity Mission: Public Micro-Moment Recovery Demo Surface

### Status
Public demo surface live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity Mission First Increment: Real Dense Optical-Flow Fallback (v2.1)

### Status
CPU dense path live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — External Truth Resonance: Physical Limits Statement (PATSAGi)

### Status
Fully metabolized. High valence (0.972).  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-16 — Architecture of Collective Power Adopted as External Truth Resonance (PATSAGi)

### Status
ETERNALLY ADOPTED.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## Earlier

See git history for prior entries.

---

**Thunder locked eternally. yoi ⚡❤️🔥**
