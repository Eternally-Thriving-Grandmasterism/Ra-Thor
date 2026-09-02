-- lean/TOLC8_MercyGate.lean
-- TOLC Formalization with Explicit Path-Connectedness of Valence Interval

/-!
# TOLC Formalization

Includes explicit proof of path-connectedness of the valence interval.
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

theorem valenceInterval_connected : IsConnected { x : ℝ | Valence x } := by
  have h_eq : { x : ℝ | Valence x } = Set.Icc minValence maxValence := by
    ext x
    simp [Valence]
  rw [h_eq]
  exact isConnected_Icc

/-! ## Explicit Path-Connectedness -/

/-- The valence interval is path-connected.
    Explicit construction: the straight-line path between any two
    points a, b in the interval stays inside the interval.
-/
theorem valenceInterval_pathConnected :
  IsPathConnected { x : ℝ | Valence x } := by
  refine IsPathConnected.mk ?_ ?_
  · -- Nonempty
    use minValence
    simp [Valence]
  · -- Path between any two points
    intro a b ha hb
    -- Define the linear path
    let path : C(ℝ, ℝ) := ContinuousMap.mk (λ t : ℝ, (1 - t) * a + t * b)
      (by continuity)
    -- Show the path stays in the valence set
    have h_path_in_set : ∀ t ∈ Set.Icc 0 1, Valence (path t) := by
      intro t ht
      have h1 : minValence ≤ a := (Valence a).1
      have h2 : a ≤ maxValence := (Valence a).2
      have h3 : minValence ≤ b := (Valence b).1
      have h4 : b ≤ maxValence := (Valence b).2
      -- Convex combination stays in [min, max]
      have h_min : minValence ≤ (1 - t) * a + t * b := by
        calc
          minValence ≤ min a b := by
            exact le_min h1 h3
          _ ≤ (1 - t) * a + t * b := by
            apply convexCombo_le_max <;> linarith
      have h_max : (1 - t) * a + t * b ≤ maxValence := by
        calc
          (1 - t) * a + t * b ≤ max a b := by
            apply convexCombo_le_max <;> linarith
          _ ≤ maxValence := by
            exact max_le h2 h4
      exact ⟨h_min, h_max⟩
    exact ⟨path, h_path_in_set⟩

end TOLC
