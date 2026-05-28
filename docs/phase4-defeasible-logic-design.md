# Technical Design Document – Phase 4: Defeasible Logic Integration

**Version:** 1.1 (Updated with Council Input)  
**Date:** May 28, 2026  
**Status:** Draft

---

## 1. Executive Overview

**Objective**  
Introduce **controlled, opt-in structural influence** of superiority and defeaters on Preferred Extensions, supported by a clear Influence Score model, context modifiers on defeaters, and strong explainability.

**Key Council-Aligned Decisions**
- Limited opt-in influence on **Preferred Extensions only**.
- Influence must be configurable, auditable, and explainable.
- Default behavior remains unchanged unless explicitly enabled.
- Context modifiers on defeaters should remain modest in Phase 4.

**Strategic Positioning**
- Phase 1–2: Built robust superiority and persistence foundations.
- Phase 3: Added expressive power through defeaters and context-aware scoring.
- Phase 4: Begins cautious expansion into **structural influence** on extensions.

---

## 2. Scope

### In Scope (Phase 4)

| Area | Description | Priority |
|------|-------------|----------|
| **Preferred Extension Influence** | Opt-in mechanisms for superiority and defeaters to affect Preferred Extensions | High |
| **Influence Score Model** | Clear, auditable scoring of how much superiority/defeaters affect arguments | High |
| **Context Modifiers on Defeaters** | Modest adjustment of defeater impact based on `SuperiorityContext` | Medium |
| **Configuration & Explainability** | Feature flags, logging, and explanation of influence | High |
| **Concrete Examples** | At least one clear re-ranking example | High |

### Out of Scope (Phase 4)

- Influence on **Stable Extensions**
- Aggressive or automatic changes to extension semantics
- Complex multi-hop propagation

---

## 3. Technical Approach

### 3.1 Primary Mechanism: Post-Filtering / Re-ranking

After computing Preferred Extensions using formal semantics, apply an optional post-processing step that re-ranks or lightly filters arguments based on superiority and defeater information.

**Key Properties**
- The core admissibility and extension computation logic remains unchanged.
- Influence is **opt-in** and configurable.
- The system can always return both the “pure formal” result and the “influenced” result.
- Influence must be explainable.

### 3.2 Influence Score Model (Core Concept)

Every argument that appears in a Preferred Extension can receive an **Influence Score** calculated from:

- Strength and direction of superiority relations (weighted by context)
- Strength and direction of defeaters (weighted by context)
- Optional context modifiers

**Example Influence Score Components**
- Strong `Council`-context superiority → Positive contribution
- High-strength defeater with `Topic` context → Negative contribution (modulated by context weight)
- Multiple defeaters → Apply diminishing returns

This score is used to re-rank or (optionally) filter arguments within the Preferred Extension results.

### 3.3 Context Modifiers on Defeaters

`SuperiorityContext` may modestly adjust the effective impact of a defeater when calculating influence on Preferred Extensions.

**Conservative Guidelines for Phase 4**
- Modifiers should be small (e.g., ±10–15%).
- `Council` context may slightly increase impact.
- Custom or mercy-oriented contexts may slightly decrease impact.
- Modifiers must be clearly documented and configurable.

### 3.4 Concrete Re-ranking Example

**Scenario**
- Two Preferred Extensions exist: `{A, B, C}` and `{A, D, E}`
- Strong superiority supports **A** (`Council` context)
- Strong defeater targets **D** (`Topic` context)

**Result after influence**
- The system may re-rank or surface `{A, B, C}` more prominently.
- It can explain: “Argument D was deprioritized due to a strong Topic-context defeater. Argument A received a boost from Council-context superiority.”

This example demonstrates both the mechanism and the required explainability.

---

## 4. Configuration & Safeguards

- All extension influence features are **opt-in**.
- The system can return both “pure formal” and “influenced” results.
- Influence must be logged and explainable.
- Strong regression tests must protect default behavior.

---

## 5. Success Criteria

Phase 4 succeeds when:

- Superiority and/or defeaters can **optionally influence Preferred Extensions** via a clear Influence Score and post-filtering/re-ranking.
- Context modifiers on defeaters are implemented modestly and documented.
- All changes are configurable, auditable, and explainable.
- Default behavior remains unchanged unless new features are explicitly enabled.
- Comprehensive tests and examples exist.

---

## 6. Implementation Roadmap (Inside Phase 4)

| Order | Focus | Notes |
|-------|-------|-------|
| 1 | Influence Score model + post-filtering | Core mechanism |
| 2 | Context modifiers on defeaters | Modest scope |
| 3 | Configuration, logging & explainability | Required |
| 4 | Concrete examples + documentation | Improves clarity |
| 5 | Testing & safeguards | Production quality |

---

## 7. Relationship to Previous Phases

| Phase | Focus | Phase 4 Builds On |
|-------|-------|-------------------|
| 1–2 | Superiority foundation + robustness | Core structures and persistence |
| 3 | Defeaters + context-aware scoring | Expressive tools that can now influence extensions |
| 4 | Structural influence on Preferred Extensions | First cautious expansion beyond scoring |

---

**End of Document**