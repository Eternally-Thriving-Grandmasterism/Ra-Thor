# Lean Mathlib Real Type Overview

## Overview

This document provides a quick reference for working with the `Real` type from Mathlib in the context of TOLC formalization (especially the `Valence` predicate).

## Import

```lean
import Mathlib.Data.Real.Basic
```

## Core Type

```lean
def Real : Type := ...
```

`Real` is Mathlib’s implementation of the real numbers. It is constructed as the Cauchy completion of the rationals and comes with a rich library of theorems.

## Key Properties Available

- **Complete ordered field**: `Real` forms a complete ordered field.
- **Decidable equality** is not available in general (reals are not decidable), but inequalities often are.
- Strong support for `linarith`, `norm_num`, `field_simp`, and `ring` tactics.

## Useful Theorems & Lemmas

### Ordering

```lean
lemma le_refl (a : ℝ) : a ≤ a
lemma le_trans {a b c : ℝ} : a ≤ b → b ≤ c → a ≤ c
lemma lt_of_le_of_ne {a b : ℝ} : a ≤ b → a ≠ b → a < b
```

### Arithmetic

```lean
lemma add_le_add {a b c d : ℝ} : a ≤ b → c ≤ d → a + c ≤ b + d
lemma mul_le_mul_of_nonneg_left {a b c : ℝ} : 0 ≤ a → b ≤ c → a * b ≤ a * c
```

### Useful Tactics

- `linarith` — Excellent for linear arithmetic over reals.
- `norm_num` — Good for normalizing numerical expressions.
- `simp [Valence]` — Useful when unfolding valence definitions.

## Relevance to TOLC Valence

Our current definition:

```lean
def Valence (x : ℝ) : Prop := minValence ≤ x ∧ x ≤ maxValence
```

This is a simple closed interval. When proving properties involving `Valence`, the following are commonly useful:

- `linarith` for interval reasoning.
- `norm_num` when dealing with concrete bounds like `0.999999`.
- `exact h` or `trivial` when the property follows directly from the definition.

## Recommendations

- Keep using `ℝ` for `Valence` as it gives access to powerful automation.
- For stricter positivity or boundedness, consider `NNReal` when appropriate.
- Document the meaning of `minValence` and `maxValence` clearly (already done).

## Related References

- `lean/TOLC8_MercyGate.lean`
- Mathlib documentation for `Data.Real.Basic`

**Mathlib’s `Real` provides excellent support for the kind of interval-based reasoning used in TOLC valence.**