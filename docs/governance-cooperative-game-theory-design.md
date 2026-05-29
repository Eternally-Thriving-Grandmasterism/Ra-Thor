# Technical Design Document – Governance & Cooperative Game Theory Layer

**Version:** 0.1 (Initial Design)  
**Date:** May 28, 2026  
**Status:** Draft  
**Related:** Phase 4 (Defeasible Logic + Context Modifiers)

---

## 1. Executive Overview

**Objective**  
Build a unified governance layer on top of Phase 4 that enables fair, explainable, and mercy-aligned decision-making using cooperative game theory (Shapley Value, Banzhaf Power Index) and structured PATSAGi Arbitration.

**Core Goals**
- Integrate Shapley Value and Banzhaf Power Index calculations.
- Create a `PatsagiArbitration` module that consumes Phase 4 `InfluenceScore` and context data.
- Maintain strong explainability and auditability.
- Keep everything opt-in and non-breaking.

---

## 2. Scope

**In Scope (Initial)**
- Data models for `GovernancePlayer` (claims, councils, agents).
- Shapley Value calculation over influence.
- Banzhaf Power Index.
- `PatsagiArbitration` that produces structured reports.
- Integration with `ArgumentGraph` and `InfluenceScore`.

**Out of Scope (Initial)**
- Full runtime council simulation.
- Persistent governance history.

---

## 3. Architecture

```
PatsagiArbitration
├── ShapleyValueCalculator
├── BanzhafPowerIndex
└── ArbitrationReport (with explanations)

↑ consumes

ArgumentGraph + InfluenceScore (Phase 4)
```

---

## 4. Key Concepts

| Concept               | Description                                      | Phase 4 Connection          |
|-----------------------|--------------------------------------------------|-----------------------------|
| GovernancePlayer      | Claim, CouncilMember, or Agent                   | `Claim` or new abstraction  |
| Value Function        | Worth of a coalition                             | `InfluenceScore.total`      |
| Shapley Value         | Fair marginal contribution                       | New module                  |
| Banzhaf Power Index   | Critical swing power                             | New module                  |
| ArbitrationReport     | Structured, explainable governance output        | New module                  |

---

## 5. Integration with Phase 4

- `PatsagiArbitration` accepts `&ArgumentGraph`.
- Uses `calculate_influence_score()` and context modifiers.
- `MercyGate` can soften negative contributions.
- `Council` context can carry higher weight.
- Outputs should be explainable (similar to `InfluenceExplanation`).

---

## 6. Phased Plan

**Phase 1**: Data models + basic Shapley calculator + tests.
**Phase 2**: Banzhaf Power Index + weighted players.
**Phase 3**: `PatsagiArbitration` module + report generation.
**Phase 4**: Rich explanations + mercy-aware heuristics + documentation.

---

## 7. Mercy & Explainability

- All results must be explainable.
- Special contexts (`MercyGate`, `Council`) should influence calculations meaningfully.
- Arbitration reports must include human-readable justification.

---

## 8. Next Steps

1. Review this design with Ra-Thor and PATSAGi Councils.
2. Begin Phase 1 implementation (foundational models + Shapley).
3. Iterate.

---

**End of Document**