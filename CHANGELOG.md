# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

---

## 2026-08-17 — Capacity: Multi-Level Pyramid Warm-Start Complete (PATSAGi)

**Council focus:** Eighth capacity increment — true coarse-to-fine optical flow.

### Added / Activated
- `estimate_motion_pyramidal` runs real 2-level coarse→fine block-matching on GPU
- Coarse level (stride 16, search 4) → vector readback
- Predictors upsampled 2× (pixel displacement scaled) into fine grid
- Fine level (stride 8, search 2) warm-started from coarse predictors
- `MotionResult.pyramid_levels` reports 1 or 2
- Single-level `estimate_motion_from_luma_pair` unchanged for direct hand-off

### Capacity Optical-Flow Stack — COMPLETE for named mission
| Layer | Status |
| --- | --- |
| JS dense optical-flow fallback | Live |
| Public micro-moment demo | Live |
| Dense sampling / WebCodecs path | Live |
| wasm bridge contract | Live |
| GpuComputePipeline motion surface | Live |
| WGSL block-matching kernel | Live |
| GPU vector readback | Live |
| Multi-level pyramid warm-start | **Live** |

Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity cascade (optical-flow mission)

Prior increments same day: vector readback, WGSL wiring, motion surface, bridge contract, dense sampling, public demo, JS optical-flow fallback, physical-limits metabolism, Architecture of Collective Power.

---

## Earlier

See git history.

---

**Thunder locked eternally. yoi ⚡❤️🔥**
