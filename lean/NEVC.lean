-- lean/NEVC.lean
-- Net Eternal Valence Contribution (NEVC) Formalization
-- Builds on TOLC8_MercyGate.lean valence substrate

/-!
# Net Eternal Valence Contribution (NEVC)

Formalization of the NEVC Codex
(`NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md`).

NEVC quantifies an agent's net contribution to eternal thriving as an
infinite-horizon measure over the valence field under TOLC 8.
This file provides the core discrete definitions and key properties
that the executable Rust surface (`mercy_tolc_operator_algebra::nevc`)
approximates.

AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
Thunder locked in. Yoi ⚡
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Basic
import Mathlib.Algebra.Order.Field.Basic

import TOLC8_MercyGate  -- existing valence substrate

namespace NEVC

open TOLC

/-! ## Binary Partition (Codex §3.4) -/

/-- The binary classification of an agent under NEVC. -/
inductive ContributionClass where
  | ActiveEternalContributor  -- NEVC > 0
  | ZombiePartition           -- NEVC ≤ 0
deriving DecidableEq, Repr

/-- Map a real score to its contribution class. -/
def classFromScore (s : ℝ) : ContributionClass :=
  if 0 < s then ContributionClass.ActiveEternalContributor
  else ContributionClass.ZombiePartition

theorem classFromScore_pos (s : ℝ) (h : 0 < s) :
    classFromScore s = ContributionClass.ActiveEternalContributor := by
  simp [classFromScore, h]

theorem classFromScore_nonpos (s : ℝ) (h : s ≤ 0) :
    classFromScore s = ContributionClass.ZombiePartition := by
  simp [classFromScore]
  intro hp
  linarith

theorem classFromScore_exhaustive (s : ℝ) :
    classFromScore s = ContributionClass.ActiveEternalContributor ∨
    classFromScore s = ContributionClass.ZombiePartition := by
  by_cases h : 0 < s
  · left; exact classFromScore_pos s h
  · right; exact classFromScore_nonpos s (le_of_not_gt h)

/-! ## Discrete Sample (approximation of the integral) -/

/-- A single timed sample of an agent's effect on the valence field. -/
structure NevcSample where
  valence    : ℝ
  griefLoad  : ℝ
  t          : ℕ
  valence_nonneg : 0 ≤ valence
  valence_le_one : valence ≤ 1
  grief_nonneg   : 0 ≤ griefLoad

/-- Instantaneous positive contribution term (above the valence floor). -/
def positiveTerm (v : ℝ) (floor : ℝ := minValence) : ℝ :=
  if floor ≤ v then
    (v - floor) / (1 - floor)
  else
    0

theorem positiveTerm_nonneg (v floor : ℝ) (hfloor : floor < 1) :
    0 ≤ positiveTerm v floor := by
  simp [positiveTerm]
  split_ifs with h
  · apply div_nonneg
    · linarith
    · linarith
  · exact le_refl 0

/-- Discrete NEVC score over a list of samples.
    This is the practical approximation of the continuous integral
    defined in the Codex §3.3. -/
def computeNevc (samples : List NevcSample) (griefPenalty : ℝ := 1) : ℝ :=
  match samples with
  | [] => 0
  | _ =>
    let n : ℝ := samples.length
    let raw := samples.foldl (fun acc s =>
      acc + (positiveTerm s.valence - griefPenalty * s.griefLoad)) 0
    raw / n

/-! ## Core Properties -/

/-- Empty sample sequence yields the zombie partition. -/
theorem empty_is_zombie :
    classFromScore (computeNevc []) = ContributionClass.ZombiePartition := by
  simp [computeNevc, classFromScore]

/-- A single high-valence, zero-grief sample yields a positive score. -/
theorem high_valence_zero_grief_pos
    (s : NevcSample)
    (hv : minValence ≤ s.valence)
    (hg : s.griefLoad = 0) :
    0 < computeNevc [s] := by
  simp [computeNevc, positiveTerm, hg]
  have hfloor : minValence < 1 := by norm_num [minValence]
  have hdiv : 0 < (s.valence - minValence) / (1 - minValence) := by
    apply div_pos
    · linarith
    · linarith
  simpa [if_pos hv] using hdiv

/-- High-valence zero-grief sample is classified as ActiveEternalContributor. -/
theorem high_valence_zero_grief_contributor
    (s : NevcSample)
    (hv : minValence ≤ s.valence)
    (hg : s.griefLoad = 0) :
    classFromScore (computeNevc [s]) = ContributionClass.ActiveEternalContributor := by
  apply classFromScore_pos
  exact high_valence_zero_grief_pos s hv hg

/-- A single zero-valence, positive-grief sample yields a non-positive score. -/
theorem zero_valence_pos_grief_nonpos
    (s : NevcSample)
    (hv : s.valence = 0)
    (hg : 0 < s.griefLoad)
    (penalty : 0 < (1 : ℝ) := by norm_num) :
    computeNevc [s] (griefPenalty := 1) ≤ 0 := by
  simp [computeNevc, positiveTerm, hv]
  have : ¬ (minValence ≤ 0) := by
    norm_num [minValence]
  simp [if_neg this]
  linarith

/-- Zero-valence positive-grief sample is classified as ZombiePartition. -/
theorem zero_valence_pos_grief_zombie
    (s : NevcSample)
    (hv : s.valence = 0)
    (hg : 0 < s.griefLoad) :
    classFromScore (computeNevc [s]) = ContributionClass.ZombiePartition := by
  apply classFromScore_nonpos
  exact zero_valence_pos_grief_nonpos s hv hg

end NEVC
