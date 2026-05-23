-- lean/TOLC8_MercyGate.lean
-- TOLC Formalization with Formal Compactness of Valence Interval

/-!
# TOLC Formalization

Includes full formal proof of compactness of the valence interval.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Topology.Instances.Real

namespace TOLC

/-! ## Valence Bounds and Predicate -/

def minValence : ℝ := 0.999999
def maxValence : ℝ := 1.0

def Valence (x : ℝ) : Prop := minValence ≤ x ∧ x ≤ maxValence

/-! ## Compactness Theorem -/

/-- The set of values satisfying `Valence` forms a closed bounded interval
    in ℝ, which is compact by the Heine-Borel theorem.

    This is one of the foundational topological properties of the
    TOLC Valence Scalar Field. -/
theorem valenceInterval_compact : IsCompact { x : ℝ | Valence x } := by
  -- Show that the set is exactly the closed interval Icc minValence maxValence
  have h_eq : { x : ℝ | Valence x } = Set.Icc minValence maxValence := by
    ext x
    simp [Valence]

  rw [h_eq]
  -- Closed bounded intervals in ℝ are compact (Heine-Borel)
  exact isCompact_Icc

end TOLC
