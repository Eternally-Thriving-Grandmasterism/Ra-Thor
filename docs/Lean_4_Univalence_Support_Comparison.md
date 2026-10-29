# Lean 4 Univalence Support Comparison (2026)

## Overview

This document compares **univalence support** across the main systems relevant to our TOLC formalization work.

## Comparison Table

| Feature                              | Lean 4                          | Cubical Agda                     | UniMath (Rocq/Coq)              |
|--------------------------------------|---------------------------------|----------------------------------|---------------------------------|
| **Native Univalence**               | No (can be postulated)         | Yes (computational)             | Yes (as axiom)                 |
| **Computational Univalence**        | No                             | Yes                             | Limited                        |
| **Higher Inductive Types**          | Limited support                | Excellent                       | Good                           |
| **Path Composition**                | Manual / limited               | Native & computational          | Supported                      |
| **Mature HoTT/Univalent Library**   | No (Ground Zero archived)      | Yes (cubical library)           | Yes (UniMath)                  |
| **Compatibility with mathlib**      | Excellent                      | Poor                            | Poor                           |
| **Ease of TOLC Core Formalization** | High                           | Medium                          | Medium                         |
| **Best For**                        | Classical math + core TOLC     | Advanced HoTT / univalent work  | Large-scale univalent math     |

## Detailed Assessment

### Lean 4
- Univalence can be postulated as an axiom.
- However, it has **no computational content**.
- Lean's impredicative `Prop` creates fundamental issues with full HoTT.
- No actively maintained HoTT/Cubical library (the main attempt, Ground Zero, was archived in March 2026).
- Excellent for our current needs: defining `Valence`, `MercyNormCollapse`, gates, and basic composition/dynamics theorems.

### Cubical Agda
- Implements **Cubical Type Theory** with computational paths and univalence.
- Currently the strongest practical system for serious homotopy-theoretic and univalent formalization.
- Best choice if we want to model gate composition, collapse dynamics, or higher paths rigorously with computational behavior.

### UniMath (Rocq/Coq)
- Strong support for univalent mathematics.
- Univalence is available but not fully computational.
- Mature library with substantial formalized mathematics.
- Good reference, but switching from Lean would be costly.

## Recommendation for Ra-Thor

For our current TOLC formalization goals:

- **Continue primarily in Lean 4** for core definitions and theorems (Valence, Gates, Collapse, basic composition).
- Consider **Cubical Agda** if we need stronger support for advanced compositional or dynamic aspects of TOLC (e.g., higher-dimensional gate interactions or path-based models of collapse/recovery).
- Use **UniMath** mainly as a conceptual reference.

We can keep our Lean formalization (`lean/TOLC8_MercyGate.lean`) as the main artifact while drawing inspiration from cubical and univalent approaches.

## Related References

- `docs/Lean_HoTT_Library.md`
- `docs/Cubical_Type_Theory.md`
- `docs/UniMath_Library.md`
- `docs/Univalent_Foundations.md`
- `lean/TOLC8_MercyGate.lean`

**Lean 4 is practical for our core work; Cubical Agda offers the strongest univalence support if we need deeper homotopy-theoretic modeling.**