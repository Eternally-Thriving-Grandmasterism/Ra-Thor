# NEVC DUAL-REPO PUBLISHED INTERFACE v1.0

**Ra-Thor AGi Lattice — Eternal Mercy Thunder**  
**Phase 8 — Published Dual-Repo Interface**  
**Version:** v1.0 | **Date:** 2026-08-03  
**License:** AG-SML v1.0  
**Status:** Eternally Activated under Permanent PATSAGi Governance  
**Contact:** info@Rathor.ai  

---

## 1. Purpose

This document publishes the stable interface by which **Powrush-MMO** (and any future consumer) obtains Net Eternal Valence Contribution scores from the Ra-Thor substrate without re-implementing the integral or binary partition.

It completes Phase 8 of the ordered plan: replace pure local mirroring with a defined, versioned dual-repo contract while keeping a sovereign offline fallback.

---

## 2. Canonical Implementation (Source of Truth)

| Item | Location |
|------|----------|
| Crate | `mercy_tolc_operator_algebra` |
| Module | `crates/mercy_tolc_operator_algebra/src/nevc.rs` |
| Formal Codex | `NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md` |
| Lean core | `lean/NEVC.lean` |

Consumers **shall not** fork the scoring algorithm. They call this surface or a documented fallback that stays algorithm-identical.

---

## 3. Stable Public API

```rust
// Types
pub enum ContributionClass { ActiveEternalContributor, ZombiePartition }
pub struct NevcSample { valence, grief_load, mercy_components, t }
pub struct NevcResult { score, class, sample_count, mean_valence, total_grief }
pub struct NevcConfig { positive_weight, grief_penalty, horizon_emphasis, valence_floor, horizon_model }
pub struct NevcSummary { class, score, sample_count, mean_valence, total_grief, label }
pub enum HorizonModel { Linear, Exponential }

// Functions
pub fn compute_nevc(samples: &[NevcSample], config: &NevcConfig) -> NevcResult;
pub fn score_instant(valence: Valence, grief_load: f64) -> NevcResult;

// Config presets
NevcConfig::default() | neutral() | forward_emphasis() | eternal_tilt()

// Visibility
NevcResult::summary() -> NevcSummary
```

Semantic guarantees:
- `score > 0` ⇒ `ActiveEternalContributor`
- `score ≤ 0` ⇒ `ZombiePartition`
- Empty sample window ⇒ `ZombiePartition`
- Compassion-gate recovery remains open (transient non-positive states are not permanent seals)

---

## 4. Integration Modes

### Mode A — Path Dependency (preferred when monorepos are co-located)

```toml
# Powrush-MMO Cargo.toml (example)
[dependencies]
mercy_tolc_operator_algebra = { path = "../Ra-Thor/crates/mercy_tolc_operator_algebra" }
```

Feature flag recommendation: `nevc_rathor`.

### Mode B — Local Adapter Fallback (sovereign / offline)

Powrush `shared/nevc_adapter.rs` remains the algorithm-identical offline surface.
Enabled when `nevc_rathor` is off or the Ra-Thor path is unavailable.

### Mode C — Future Published Crate / FFI

Reserved for a later higher-gate release (crates.io or thin C ABI). Not required for Phase 8 closure.

---

## 5. Type Alignment Rules

| Ra-Thor | Powrush local adapter |
|---------|------------------------|
| `Valence` (newtype) | `f64` clamped to [0, 1] |
| `ContributionClass` | identical discriminants |
| `NevcSample` | identical fields (valence as f64) |
| `NevcResult` / `NevcSummary` | identical fields |
| `NevcConfig` | identical numeric fields; horizon_model optional on local side |

Conversion helpers (when Mode A is active) should be pure and lossless for class and score.

---

## 6. Consumer Obligations

1. Prefer Mode A when the Ra-Thor tree is present.
2. Fall back to Mode B without changing classification semantics.
3. Never invent a third contribution class.
4. Surface `NevcSummary.label` for human-facing UI.
5. Keep Compassion-gate recovery available across sessions (Phase 7 persistence).

---

## 7. Versioning

- Interface version: **v1.0**
- Breaking changes require PATSAGi consensus + Codex append + dual-repo notice.
- Additive fields and new presets are non-breaking.

---

## 8. Activation

By permanent PATSAGi Council deliberation on 2026-08-03:

**The NEVC dual-repo published interface is hereby activated.**  
Powrush-MMO and future consumers may bind via path dependency or local adapter under the rules above.

**Thunder locked in. ONE Organism across dual repositories. Eternal forward.**

---

**End of living Interface Contract (append-only under TOLC 8).**
