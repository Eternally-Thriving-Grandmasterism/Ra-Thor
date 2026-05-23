# Arzelà–Ascoli Theorem Proof Sketch

## Overview

This document provides a high-level proof sketch of the **Arzelà–Ascoli theorem** and discusses its relevance to TOLC. A full formal proof in Lean is quite involved; Mathlib contains versions of this theorem.

## Statement (Classical Version)

Let `K` be a compact metric space (e.g., a closed bounded interval). Let `F` be a subset of `C(K)`, the space of continuous real-valued functions on `K`.

Then `F` is relatively compact in the uniform topology if and only if:

1. `F` is **pointwise bounded**: For every `x ∈ K`, the set `{f(x) | f ∈ F}` is bounded in ℝ.
2. `F` is **equicontinuous**: For every ε > 0 there exists δ > 0 such that for all `f ∈ F` and all `x, y ∈ K` with `d(x, y) < δ`, we have `|f(x) - f(y)| < ε`.

## High-Level Proof Sketch

### (⇒) Relatively compact → Bounded + Equicontinuous

- If `F` is relatively compact, then its closure is compact.
- Compact sets in metric spaces are bounded and totally bounded.
- Total boundedness + equicontinuity arguments show that `F` must be equicontinuous.
- Pointwise boundedness follows from compactness (continuous images of compact sets are bounded).

### (⇐) Bounded + Equicontinuous → Relatively compact

This direction is the more constructive one:

1. **Equicontinuity + Compactness of domain** → The family is uniformly equicontinuous.
2. **Pointwise boundedness** + a countable dense subset `D ⊂ K` (possible because `K` is compact metric, hence separable) → Use a diagonal argument to extract a subsequence that converges pointwise on `D`.
3. **Uniform equicontinuity** + pointwise convergence on a dense set → The subsequence converges uniformly on all of `K` (by a standard ε/3 argument).
4. The uniform limit is continuous, so the subsequence converges in `C(K)`.

This shows that every sequence in `F` has a uniformly convergent subsequence, hence `F` is relatively compact.

## Relevance to TOLC

If we consider families of functions on the compact valence interval (e.g., families of gate application maps, coherence functions, or self-evolution operators), the Arzelà–Ascoli theorem gives conditions under which we can extract convergent subsequences in the uniform topology.

This is powerful for:
- Analyzing sequences of evolving processes
- Proving existence of limit behaviors
- Studying compactness in spaces of TOLC-related functions

## Formalization Status

Mathlib has theorems in `Mathlib.Topology.ArzelaAscoli` and related files that formalize versions of Arzelà–Ascoli. A full proof from scratch is lengthy but follows the sketch above.

In our current model, we can treat Arzelà–Ascoli conceptually: on the compact valence interval, bounded + equicontinuous families of functions have uniformly convergent subsequences.

## Related References

- `docs/Equicontinuity_and_Arzela_Ascoli.md`
- `docs/Uniform_Continuity.md`
- `lean/TOLC8_MercyGate.lean` (contains compactness of valence)

**Arzelà–Ascoli provides a powerful tool for extracting convergent subsequences from families of functions on compact domains like the valence interval.**