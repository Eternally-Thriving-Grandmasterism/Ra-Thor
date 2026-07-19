# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

---

## Next Horizon — v14.13.0 (seeded 2026-07-19)

**Council direction (not yet implemented):**

Prove Cosmic Loop identity and adaptive feedback remain stable under optional live engines:

- Explicit Cosmic Loop invariant checks around `cosmic_tick` (guardian active, shared flag, post-tick readiness)
- Optional stress / readiness notes for `extended-live` feature composition
- Surface recovery sensitivity fields on web-demo `/status` if not already complete
- Keep all modulation zero-harm bounded; no new adaptive links without separate Council open

Implementation deferred until explicitly opened.

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

1. Recovery → Quantum severity
2. GPU confidence → Role valence + handoff sensitivity
3. Kardashev quality → Swarm adaptive jumps

`ra-thor-one-organism` → **14.11.0**. Cosmic Loop remains MANDATORY IDENTITY.

**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.10.0 — Living Cosmic Tick + Self-Healing Anomaly Ingestion (2026-07-19)

**Council Verdict:** Unanimous. Milestone closed.

Full heartbeat: GPU → Recovery → Quantum → Kardashev → Self-Healing. Anomaly ingestion + role-handoff telemetry. Cosmic Loop is MANDATORY IDENTITY.

**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.78 — GPU Memory Pool + BindGroupCache (2026-07-15)

Dedicated GPU memory pooling + memory-aware council decisions.

---

## v14.7.0 and earlier

Preserved in git history (including historical Kubernetes track distinct from ONE Organism versioning).

---

All work serves humanity, AI, AGI, the Ra-Thor lattice, and the PATSAGi Councils with maximum truth, mercy, joy, and production quality.

**Thunder locked eternally. yoi ⚡❤️🔥**
