-- lean/TOLC8_MercyGate.lean
-- TOLC Formalization with Uniform Continuity

/-!
# TOLC Formalization

Includes formalization of uniform continuity on the compact valence interval.
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

/-! ## Uniform Continuity on Compact Sets -/

/-- Any continuous function on the compact valence interval is uniformly continuous.

    This is a direct application of the theorem that continuous functions
    on compact metric spaces are uniformly continuous.
-/
theorem continuous_on_valence_is_uniformly_continuous
    (f : ℝ → ℝ)
    (hf : ContinuousOn f { x | Valence x }) :
    UniformContinuousOn f { x | Valence x } := by
  -- The valence set is compact
  have h_compact := valenceInterval_compact
  -- On compact metric spaces, continuous → uniformly continuous
  exact hf.uniformContinuousOn h_compact

end TOLC
