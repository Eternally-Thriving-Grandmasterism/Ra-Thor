-- lean/TOLC8_MercyGate.lean
-- TOLC Formalization with Connectedness of Valence Interval

/-!
# TOLC Formalization

Includes proof that the valence interval is connected.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Topology.Instances.Real

namespace TOLC

/-! ## Valence Bounds and Predicate -/

def minValence : ℝ := 0.999999
def maxValence : ℝ := 1.0

def Valence (x : ℝ) : Prop := minValence ≤ x ∧ x ≤ maxValence

/-! ## Compactness -/

theorem valenceInterval_compact : IsCompact { x : ℝ | Valence x } := by
  have h_eq : { x : ℝ | Valence x } = Set.Icc minValence maxValence := by
    ext x
    simp [Valence]
  rw [h_eq]
  exact isCompact_Icc

/-! ## Connectedness -/

/-- The set of values satisfying `Valence` forms a closed interval,
    which is connected in ℝ.

    In fact, it is path-connected and convex.
    This is useful for interpolation arguments and continuity-based
    reasoning within the valid valence range.
-/
theorem valenceInterval_connected : IsConnected { x : ℝ | Valence x } := by
  have h_eq : { x : ℝ | Valence x } = Set.Icc minValence maxValence := by
    ext x
    simp [Valence]
  rw [h_eq]
  exact isConnected_Icc

end TOLC
