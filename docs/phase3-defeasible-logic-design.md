# Technical Design Document – Phase 3: Defeasible Logic Integration

**Version:** 2.0 (Production-Grade, Enriched)  
**Date:** May 28, 2026  
**Status:** Finalized Direction after Council Deliberation

---

## 1. Executive Overview

### Objective
Extend Ra-Thor’s defeasible reasoning system by introducing a first-class **Defeater** primitive and meaningful context-aware scoring. This significantly increases the system’s ability to model nuanced, mercy-aligned deliberation while preserving the stability and auditability of formal extension computation.

### Strategic Positioning
- **Phase 1**: Established basic superiority relations and their influence on recommendation scoring.
- **Phase 2**: Added conflict resolution (recency wins), opt-in persistence, and the `SuperiorityContext` enum.
- **Phase 3**: Introduces **defeaters** as a distinct weakening mechanism and enhances scoring with context. This is the most expressive layer so far without modifying core extension algorithms.
- **Phase 4 (Future)**: Will explore deeper influence on Preferred/Stable Extensions and context-based modifiers on defeater strength.

### Design Philosophy
- **Professional-grade clarity and auditability** — Every new concept must be well-defined and testable.
- **Mercy-aligned expressiveness** — Support gentle, context-sensitive weakening without harsh invalidation.
- **Truth-seeking stability** — Core formal semantics (Grounded, Preferred, Stable) remain untouched.
- **Production readiness** — Clean APIs, clear separation of concerns, and comprehensive testing.

---

## 2. Scope

### In Scope (Phase 3)

| Area | Description | Quality Standard |
|------|-------------|------------------|
| **Defeater Primitive** | First-class `Defeater` with independent strength and smart defaults | Production-grade modeling |
| **Defeater Integration** | Defeaters reduce target credibility in the Recommendation Engine | Clean, testable integration |
| **Context-Aware Scoring** | `SuperiorityContext` meaningfully influences Safety and Evolution scores | Professional, tunable design |
| **PATSAGi Archetype Hooks** | Lightweight, optional linking of defeaters/superiority to council types | Clean and optional |
| **Testing & Documentation** | Comprehensive tests + clear conceptual documentation | High coverage + professional docs |

### Explicitly Out of Scope (Deferred to Phase 4)

- Any modification to **Grounded, Preferred, or Stable Extension** computation.
- Context-based modifiers on individual defeater strength.
- Automatic inference or learning of defeaters/superiority.
- Complex multi-defeater propagation or chaining logic.

---

## 3. Defeater Model (Production Specification)

### 3.1 Core Structure

```rust
#[derive(Debug, Clone)]
pub struct Defeater {
    pub id: ArgumentId,
    pub source_claim_id: ArgumentId,
    pub target_claim_id: ArgumentId,
    pub strength: f64,                    // 0.0 – 1.0
    pub provided_by: String,
    pub context: Option<SuperiorityContext>,
}
```

### 3.2 Strength Rules (Finalized)

| Situation | Behavior | Rationale |
|-----------|----------|-----------|
| Explicit strength provided | Use the provided value | Maximum control and flexibility |
| No strength provided | Default = source claim’s `strength` | Intuitive and simple by default |
| Context modifiers on strength | **Deferred to Phase 4** | Keep Phase 3 focused and stable |

### 3.3 Semantic Meaning

A **Defeater** represents a reason why a claim should be treated with **reduced confidence or weight**, without fully attacking or invalidating it. It is conceptually “softer” than an `Attack`.

- **Attack** → Challenges validity or truth of a claim.
- **Defeater** → Reduces the *practical weight* or *credibility* of a claim in deliberation.

This distinction is important for mercy-aligned reasoning (e.g., “This claim is true, but we should give it less weight in this context due to X”).

---

## 4. Integration Points

### 4.1 Recommendation Engine

Defeaters will reduce the **effective strength** or **credibility** of the target claim when calculating:
- Safety Score
- Evolution Potential
- Overall Recommendation

**Accumulation Rule (Proposed):**
Multiple defeaters can accumulate, but with **diminishing returns** to prevent over-punishment of a claim.

### 4.2 Proposed API Additions

```rust
pub fn add_defeater(
    &mut self,
    source_claim_id: ArgumentId,
    target_claim_id: ArgumentId,
    strength: Option<f64>,
    provided_by: String,
    context: Option<SuperiorityContext>,
) -> Option<ArgumentId>;

pub fn get_defeaters(&self, claim_id: ArgumentId) -> Vec<&Defeater>;
```

### 4.3 Persistence

Defeaters will follow the same **opt-in persistence** pattern established in Phase 2 for superiority relations (stored as JSON in SQLite).

---

## 5. Context-Aware Scoring (Phase 3)

### 5.1 Goals
- Allow `SuperiorityContext` variants to carry **differentiated influence** on scoring.
- Make recommendation output more context-sensitive and council-aware.

### 5.2 Example Behavior (Phase 3)

| Context | Influence on Scoring | Notes |
|---------|----------------------|-------|
| `Council` | Slightly higher weight | Reflects institutional/structural importance |
| `Topic` | Domain-specific weighting | Useful for topic-focused deliberations |
| `General` | Baseline behavior | Default neutral influence |
| `Custom(...)` | User-defined (future) | Extensibility hook |

**Note:** Context-based modifiers on *defeater strength itself* are deferred to Phase 4.

---

## 6. PATSAGi Council Integration (Lightweight)

### 6.1 Optional Archetype Hooks

Provide lightweight, optional mechanisms so that certain council types can naturally interact with defeaters and superiority:

- Example: Mercy-oriented councils can more easily act as defeaters in contexts involving compassion or redemption.
- These hooks should be **opt-in** and clearly documented.

### 6.2 Design Constraint
Keep integration lightweight in Phase 3. Avoid creating heavy new gate machinery or complex archetype logic.

---

## 7. Risks & Mitigations (Production Grade)

| Risk | Mitigation |
|------|------------|
| Conceptual confusion between Attack vs Defeater | Clear documentation + distinct APIs + semantic explanation |
| Over-weakening from many defeaters | Apply diminishing returns in scoring logic |
| Scope creep into extension algorithms | Strict architectural boundaries and explicit “Out of Scope” list |
| Complexity growth | Phased delivery + strong regression tests + professional documentation |

---

## 8. Success Criteria

Phase 3 is successful when:

- A clean, well-documented `Defeater` primitive exists with independent strength and smart defaults.
- Defeaters correctly and meaningfully influence recommendation scoring.
- Context-aware scoring using `SuperiorityContext` is implemented and testable.
- The system remains stable, auditable, and aligned with Ra-Thor’s mercy and evolution principles.
- All existing Phase 1 and Phase 2 functionality continues without regression.
- Documentation is production-grade (clear concepts, examples, and rationale).

---

## 9. Implementation Roadmap (Inside Phase 3)

| Order | Focus | Deliverable | Quality Notes |
|-------|-------|-------------|---------------|
| 1 | Defeater primitive | `Defeater` struct + `add_defeater()` | Clean API, good defaults |
| 2 | Scoring integration | Defeaters affect Safety/Evolution scores | Diminishing returns, testable |
| 3 | Context-aware scoring | `SuperiorityContext` influences scores | Professional, tunable |
| 4 | Persistence | Opt-in SQLite support for defeaters | Follows Phase 2 pattern |
| 5 | Tests + Documentation | Full regression suite + enriched design doc | Production quality |

---

## 10. Relationship to Previous Phases

| Phase | Focus | How Phase 3 Builds On It |
|-------|-------|--------------------------|
| 1 | Basic superiority + strict claims + scoring impact | Foundation for weakening mechanisms |
| 2 | Conflict resolution, persistence, context typing | Robustness + `SuperiorityContext` enum |
| 3 | Defeaters + context-aware scoring + council hooks | New expressive layer on top of stable base |

Phase 3 significantly increases the system’s ability to model real-world deliberative reasoning while protecting the integrity of formal semantics.

---

## 11. Open Questions (Minor – Can Be Resolved During Implementation)

1. Should multiple defeaters on the same claim use simple summation or a more sophisticated accumulation model?
2. How should the system visually or semantically distinguish “defeated but still valid” claims in council outputs?

These are considered minor and can be decided during implementation.

---

**End of Document**