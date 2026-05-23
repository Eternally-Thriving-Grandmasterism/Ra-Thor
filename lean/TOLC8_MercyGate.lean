-- lean/TOLC8_MercyGate.lean
-- TOLC Formalization with Compactness of Valence Interval

/-!
# TOLC Formalization

Includes proof of compactness of the valence interval.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Topology.Instances.Real

namespace TOLC

/-! ## Valence Bounds -/

def minValence : ℝ := 0.999999
def maxValence : ℝ := 1.0

/-! ## Valence Predicate -/

def Valence (x : ℝ) : Prop := minValence ≤ x ∧ x ≤ maxValence

/-! ## Compactness of Valence Interval -/

/-- The set of values satisfying Valence is a closed bounded interval,
    hence compact in ℝ (by Heine-Borel theorem).

    This justifies strong topological properties:
    - Every sequence in the interval has a convergent subsequence.
    - Continuous functions attain max/min on the interval.
    - The interval is sequentially compact.
-/
theorem valenceInterval_compact :
  IsCompact { x : ℝ | Valence x } := by
  -- The set is exactly the closed interval Icc minValence maxValence
  have h_eq : { x : ℝ | Valence x } = Set.Icc minValence maxValence := by
    ext x
    simp [Valence]
  rw [h_eq]
  -- Mathlib knows that closed bounded intervals in ℝ are compact
  exact isCompact_Icc

end TOLC
