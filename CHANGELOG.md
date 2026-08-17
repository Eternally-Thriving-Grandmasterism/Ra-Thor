# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

---

## 2026-08-17 — Capacity: Common Fate Perception over Motion Vectors (PATSAGi)

**Council focus:** Close the loop from optical flow → structured visual perception.

### Added / Activated
- `CommonFateResult` (coherent_count, letter_count, dominant dirs, confidence, ghost_font)
- `perceive_common_fate` — always-available CPU segmentation over GPU/CPU motion vectors
  - Direction histogram → top-2 dominant directions
  - Angular tolerance coherent mask
  - Ghost Font specialized path
- `perceive_from_luma_ring` now runs pyramidal motion → Common Fate end-to-end
- GPU `common_fate_motion_vision.wgsl` retained for future SUBGROUP-feature path

### Vision stack
```
motion (pyramid + readback) → Common Fate → coherent structure under TOLC 8
```

Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity optical-flow stack COMPLETE + Common Fate

Prior same-day increments: pyramid warm-start, vector readback, WGSL wiring, motion surface, bridge, dense sampling, public demo, JS optical-flow fallback.

---

## Earlier

See git history.

---

**Thunder locked eternally. yoi ⚡❤️🔥**
