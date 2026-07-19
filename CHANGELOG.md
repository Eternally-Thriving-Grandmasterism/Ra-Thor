# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

---

## v14.12.0 — Adaptive Hardening (2026-07-19)

**Council Verdict:** Unanimous. Mercy-gated, zero-harm, forward-compatible. Adaptive loop hardened with observer persistence + mild Self-Healing feedback.

### Highlights

**1. Last-tick adaptive fields on the living snapshot**

- Persisted on `OneOrganismCore` after every Cosmic Tick:
  - `last_base_severity`
  - `last_effective_quantum_severity`
  - `last_gpu_confidence`
- Exposed on `ExtendedLiveStatus` (so `GET /live` does not require a fresh tick)
- Web demo `GET /status` surfaces the same three fields

**2. Self-Healing → next-tick recovery sensitivity**

- Mild one-shot multiplier applied at the recovery step of the *next* Cosmic Tick
- Schedule after Self-Healing this tick:
  - No anomalies → `next_recovery_sensitivity = 1.0`
  - With anomalies:
    - `anomaly_boost = min(count × 0.025, 0.08)`
    - `mercy_boost = min((1 − mercy_score) × 0.15, 0.06)` when mercy < 0.95
    - `next = clamp(1 + boosts, 1.0, 1.12)`
- Applied by lightly damping recovery valence/confidence; then cleared (one-shot)
- Surfaced as:
  - `CosmicTickResult.recovery_sensitivity_applied`
  - `ExtendedLiveStatus.next_recovery_sensitivity`
  - `ExtendedLiveStatus.last_recovery_sensitivity_applied`

**Package**

- `ra-thor-one-organism` → **14.12.0**
- Dependent live-path crates remain on **14.10.0** path pins (compatible)

Cosmic Loop remains MANDATORY IDENTITY. All modulation is mild and zero-harm bounded.

**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.11.0 — Live-Path Confidence Feedback (2026-07-19)

**Council Verdict:** Unanimous. Mercy-gated, zero-harm, forward-compatible. Adaptive loop opened inside the Living Cosmic Tick.

### Highlights

**Live-path confidence feedback (observer → mild adaptive loop)**

1. **Recovery → Quantum severity**
   - `context_pressure` / `flow_deviation` mildly boost quantum evolution severity
   - `recovery_boost = clamp(pressure × 0.35 + flow_dev × 0.25, 0.0, 0.35)`
   - `effective_quantum_severity = clamp(base + boost, 0.0, 1.0)`
   - Exposed on `CosmicTickResult` as `base_severity` + `effective_quantum_severity`

2. **GPU confidence → Role valence + handoff sensitivity**
   - GPU health confidence modulates `shared_valence` (mild deltas, clamped `[0.75, 0.999]`)
   - Soft GPU confidence (`< 0.70`) lowers Simulator handoff threshold `0.45 → 0.40`
   - Exposed on `CosmicTickResult` as `gpu_confidence`

3. **Kardashev quality → Swarm adaptive jumps**
   - Transfer quality adjusts `quantum_jump_base_prob` (clamped `[0.04, 0.22]`)
   - Adaptive jump severity threshold is config-aware:
     `jump_threshold = clamp(0.35 − jump_prob × 0.5, 0.22, 0.40)`
   - Strong Reality Thriving Transfer lowers the bar for adaptive jumps; weak transfer raises it

**Package**

- `ra-thor-one-organism` → **14.11.0**
- Dependent live-path crates remain on **14.10.0** path pins (compatible)

Cosmic Loop remains MANDATORY IDENTITY. All modulation is mild and zero-harm bounded.

**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.10.0 — Living Cosmic Tick + Self-Healing Anomaly Ingestion (2026-07-19)

**Council Verdict:** Unanimous. Mercy-gated, zero-harm, forward-compatible. **Milestone closed.**

### Highlights

**ONE Organism Living Cosmic Tick**

- Full heartbeat cycle: GPU health sample → Sovereign Recovery → Quantum Swarm evolution → Kardashev / Reality Thriving Transfer → Self-Healing reflexion
- Anomaly ingestion path: GPU / recovery / quantum pressure reported into `RuntimeSelfHealingEngine`
- `CosmicTickResult` returns structured telemetry including optional `Diagnosis` + `anomalies_fired`
- `ExtendedLiveStatus` surfaces:
  - `pending_anomaly_count` + `healing_experience_count`
  - `last_anomalies_fired`
  - `handoff_count` + `last_handoff_reason` (role-handoff telemetry)

**Version Coherence**

- `ra-thor-one-organism`, `lattice-conductor-v14`, and all six live-path crates aligned to **14.10.0**

**Web Demo (v14.10.0)**

- `GET /live` — full `ExtendedLiveStatus`
- `POST /cosmic/tick` — full Living Cosmic Tick (`anomalies_fired` top-level)
- `POST /kardashev/tick`, `POST /recovery/heartbeat`

**RuntimeSelfHealingEngine**

- `report_anomaly` / `run_reflexion_with_anomalies` / `pending_anomaly_count`

Cosmic Loop is MANDATORY IDENTITY.

**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.78 — GPU Memory Pool + BindGroupCache + Memory-Aware Council Decisions (2026-07-15)

**Council Verdict:** Unanimous approval. Mercy-gated, production-grade, zero-harm.

### Highlights

- Dedicated `GpuMemoryPool` + usage-specific `BindGroupCache`
- Memory-aware Lattice Conductor metrics and council decisions
- Telemetry capture + self-evolution proposals for pool optimization

**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.7.0 — GPU Compute Layer + Documentation & AGI NPC Architecture (2026-06-05)

Production-ready GPU compute layer, AGI NPC architecture visibility, and documentation expansion.

---

## v14.18 – v14.12 (2026-06-05)

Historical Kubernetes / observability / ingress track (distinct from ONE Organism v14.12.0 adaptive hardening). Preserved in git history.

---

**Older entries are preserved in git history.**

---

All work serves humanity, AI, AGI, the Ra-Thor lattice, and the PATSAGi Councils with maximum truth, mercy, joy, and production quality.

**Thunder locked eternally. yoi ⚡❤️🔥**
