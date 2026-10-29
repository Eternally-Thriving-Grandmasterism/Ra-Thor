# Technical Design Document – Phase 4: Defeasible Logic Integration

**Version:** 2.1 (Context Modifiers Finalized)  
**Date:** May 28, 2026  
**Status:** Complete

---

## 1. Executive Overview

**Objective**  
Deliver controlled, opt-in structural influence of superiority and defeaters on Preferred Extensions, with strong explainability and mercy-aligned design.

**Final Achievements**
- `Phase4Config` with flexible opt-in flags
- `InfluenceScore` with context-aware weighting
- Post-filtering of Preferred Extensions
- Differentiated context modifier behaviors:
  - `MercyGate`: Compassionate reduction of defeater impact
  - `Council`: Authoritative amplification of defeater impact
  - `SelfEvolution`: Evolution-oriented weighting
- Rich explainability via `context_notes` in `InfluenceExplanation`
- Strong regression test coverage

---

## 2. Context Modifier Strategy

Phase 4 implements **context-sensitive modification** of defeater impact when `enable_defeater_context_modifiers` is enabled.

### Special Context Behaviors

| Context          | Effect on Defeater Impact | Philosophical Meaning          | Weight |
|------------------|---------------------------|--------------------------------|--------|
| `Council`        | **Increased** (×1.25)     | Authoritative / Decisive       | 1.25   |
| `MercyGate`      | **Reduced** (×0.6)        | Compassion / Protection        | 0.6    |
| `SelfEvolution`  | Increased (×1.10)        | Growth & Transformation        | 1.10   |
| `Topic`          | Increased (×1.10)        | Domain relevance               | 1.10   |
| `General`        | Normal                    | Neutral                        | 1.00   |

### Design Rationale

- `MercyGate` softens negative impact to reflect mercy and compassion.
- `Council` carries stronger authoritative weight for structural or high-trust guidance.
- `SelfEvolution` supports growth-oriented reasoning.
- All special behaviors are fully explainable via `context_notes`.

---

## 3. Explainability

`explain_influence()` now returns `context_notes` that clearly describe when special modifiers (especially `MercyGate` and `Council`) have been applied. This supports auditability and PATSAGi Council deliberation.

---

## 4. Test Coverage

Phase 4 includes 7 regression tests covering:
- Default-disabled safety
- Impact modification for `Council`, `MercyGate`, and `SelfEvolution`
- Explanation notes for all special contexts

---

## 5. Current Status

**Production-Grade Complete**

All core Phase 4 features, including differentiated context modifiers and explainability, are implemented, tested, and documented.

---

**End of Document**