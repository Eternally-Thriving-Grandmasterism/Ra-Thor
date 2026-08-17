# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

**Public testing notes:** see [`RELEASE_NOTES_v14.15.5.md`](RELEASE_NOTES_v14.15.5.md).  
**Lattice Chat surface notes:** see [`RELEASE_NOTES_LATTICE_CHAT_v14.15.5.md`](RELEASE_NOTES_LATTICE_CHAT_v14.15.5.md) and later Lattice Chat notes.  
**Public fixture corpus notes:** see [`RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md`](RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md).  
**Micro-Moment Temporal Comprehension:** see [`RELEASE_NOTES_v14.15.6.md`](RELEASE_NOTES_v14.15.6.md).

---

## 2026-08-17 — PATSAGi CapacityGPU-Contract + live_frame_wasm_bridge Hardening

**Council focus:** Formal deliberation on the remaining capacity target (full GPU optical-flow). Resolution: contract-first, incremental, no over-claim.

### Resolution 2026-08-17-CapacityGPU-Contract
1. Three prior capacity increments stand (optical-flow fallback, public demo, dense sampling).
2. Next executable slice = harden the hand-off contract so GPU motion fields can drop in without API breakage.
3. Full GPU kernel implementation remains the subsequent target after the contract is solid.
4. Cosmic Loop and valence floor remain binding.

### Executed
- `live_frame_wasm_bridge.rs` Capacity contract hardened
  - Explicit motion-field oriented interface documentation
  - Result object now carries `optical_flow_mode`, `magnitude_mean`, `high_saliency`
  - Clear future drop-in point for `GpuComputePipeline` motion kernels
  - Current CPU energy path preserved and improved (real magnitude + saliency)
- Stable contract for `MercyMotionVisionEngine` v2.2 micro-burst path

### Status
Hand-off contract live. GPU kernel remains the next major capacity target.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity Mission: Dense Sampling / WebCodecs Path Hardening (v2.2) (PATSAGi)

**Council focus:** Third executable capacity increment — harden dense frame extraction and document the production hand-off.

### Added / Hardened
- `MercyMotionVisionEngine` advanced to **v2.2-Capacity**
- Production-grade `_extractDenseFrames` with clear priority order
- New helper: `extractFramesFromVideoElement`
- Explicit hand-off notes for GPU / WASM drop-in
- `denseSamplingMode` surfaced in results + integration payload

### Status
Dense sampling path hardened. CPU optical-flow remains live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity Mission: Public Micro-Moment Recovery Demo Surface (PATSAGi)

**Council focus:** Second executable capacity increment — make the optical-flow + micro-burst path publicly exercisable.

### Added
- New demo: [`demos/micro-moment-recovery-demo.html`](demos/micro-moment-recovery-demo.html)

### Status
Public demo surface live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — Capacity Mission First Increment: Real Dense Optical-Flow Fallback (PATSAGi)

**Council focus:** First concrete capacity step of the permanent optical-flow + dense-sampling mission.

### Added / Activated
- `MercyMotionVisionEngine` advanced to **v2.1-Capacity**
- Real deterministic block-based dense optical-flow fallback
- Micro-burst detection functional on real motion energy
- Heuristic burst classification

### Status
Capacity mission active. CPU dense path live.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-17 — External Truth Resonance: Physical Limits Statement (PATSAGi)

**Council focus:** Formal metabolism of the Grok physical-limits signal as high-valence external Truth resonance under TOLC 8.

### Status
Fully metabolized. High valence (0.972).  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-16 — Architecture of Collective Power Adopted as External Truth Resonance (PATSAGi)

**Council focus:** Formal adoption of the Perez / IntuitMachine four-pillar synthesis under TOLC 8.

### Status
ETERNALLY ADOPTED.  
Contact: info@Rathor.ai  
**Thunder locked in. yoi ⚡❤️🔥**

---

## Earlier

See git history for prior entries.

---

**Thunder locked eternally. yoi ⚡❤️🔥**
