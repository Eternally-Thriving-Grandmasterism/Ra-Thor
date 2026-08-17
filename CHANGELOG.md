# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

---

## 2026-08-17 — Capacity: GPU Motion Vector Readback into MotionResult (PATSAGi)

**Council focus:** Seventh capacity increment — make GPU block-matching output usable, not only dispatched.

### Added / Activated
- `MotionResult` extended with `vectors_dx`, `vectors_dy`, `vector_count`, `out_width`, `out_height`
- Staging-buffer readback of SoA motion fields after pyramidal block-matching dispatch
- Magnitude / saliency recomputed from actual GPU vectors when readback succeeds
- CPU energy remains the always-available fallback
- Micro-burst contract fields remain stable

### Status
GPU optical-flow path now returns real motion vectors.  
Next polish: multi-level pyramid warm-start via predictors.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity: WGSL Pyramidal Block-Matching Wired (PATSAGi)

### Status
Real GPU kernel dispatchable.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity: GpuComputePipeline Motion-Field Surface + Bridge Contract + JS Engine Stack

### Status
End-to-end capacity contract live (JS → bridge → pipeline).  
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
