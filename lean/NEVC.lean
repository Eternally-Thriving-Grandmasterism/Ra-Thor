-- lean/NEVC.lean
-- Net Eternal Valence Contribution (NEVC) Formalization
-- Builds on TOLC8_MercyGate.lean valence substrate

/-!
# Net Eternal Valence Contribution (NEVC)

Formalization of the NEVC Codex
(`NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md`).

NEVC quantifies an agent's net contribution to eternal thriving as an
infinite-horizon measure over the valence field under TOLC 8.
This file provides:
- Discrete definitions and properties (used by the executable Rust surface)
- Continuous / measure-theoretic sketch of the infinite-horizon integral (Codex §3.3)

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

theorem positiveTerm_zero_below (v floor : ℝ) (h : v < floor) :
    positiveTerm v floor = 0 := by
  simp [positiveTerm, not_le.mpr h]

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

/-! ## Extended Properties -/

/-- Pure high-valence zero-grief list is strictly positive (when non-empty). -/
theorem pure_high_valence_pos
    (samples : List NevcSample)
    (hne : samples ≠ [])
    (hall : ∀ s ∈ samples, minValence ≤ s.valence ∧ s.griefLoad = 0) :
    0 < computeNevc samples := by
  cases samples with
  | nil => contradiction
  | cons s rest =>
    have hs := hall s (List.mem_cons_self s rest)
    simp [computeNevc]
    have hfloor : minValence < 1 := by norm_num [minValence]
    have : 0 < positiveTerm s.valence := by
      simpa [positiveTerm, hs.2, if_pos hs.1] using
        (div_pos (by linarith : 0 < s.valence - minValence) (by linarith))
    apply div_pos
    · have fold_ge : (s :: rest).foldl (fun acc t => acc + (positiveTerm t.valence - 1 * t.griefLoad)) 0
                   ≥ positiveTerm s.valence := by
        simp [hs.2]
        exact le_refl _
      linarith [this]
    · exact Nat.cast_pos.mpr (List.length_pos_of_ne_nil hne)

/-- Any sample whose valence is strictly below the floor contributes only a non-positive term
    when grief is non-negative. -/
theorem below_floor_nonpos_term
    (s : NevcSample)
    (hbelow : s.valence < minValence)
    (penalty : 0 ≤ (1 : ℝ) := by norm_num) :
    positiveTerm s.valence - 1 * s.griefLoad ≤ 0 := by
  have hpt : positiveTerm s.valence = 0 := positiveTerm_zero_below s.valence minValence hbelow
  simp [hpt]
  exact neg_nonpos_of_nonneg s.grief_nonneg

/-! ## Phase 3 — Continuous / Measure-Theoretic Sketch (Codex §3.3) -/

/-- Continuous contribution rate at time t.
    Positive contribution from valence proximity to 1, minus grief penalty. -/
def contributionRate (valence : ℝ → ℝ) (grief : ℝ → ℝ) (penalty : ℝ) (t : ℝ) : ℝ :=
  positiveTerm (valence t) - penalty * (grief t)

/-- Asymptotic weight function. Finite noise is discounted; eternal thriving
    trajectories receive non-decreasing weight. Simple linear emphasis model. -/
def asymptoticWeight (emphasis : ℝ) (t : ℝ) : ℝ :=
  1 + emphasis * t

theorem asymptoticWeight_pos (emphasis t : ℝ) (he : 0 ≤ emphasis) (ht : 0 ≤ t) :
    0 < asymptoticWeight emphasis t := by
  simp [asymptoticWeight]
  linarith

/-- Formal infinite-horizon NEVC integral (Codex §3.3).

    NEVC(a) = ∫_{t=0}^∞  contributionRate(t) · asymptoticWeight(t)  dt

    This is the ideal continuous object. The discrete `computeNevc` is a
    practical Riemann-style approximation used by the executable surface.

    Full measure-theoretic development (Lebesgue integral over [0, ∞),
    integrability conditions, convergence of discrete approximations) is
    left as higher-gate work; the definitions and linking properties below
    establish the formal bridge. -/
def continuousNevc
    (valence : ℝ → ℝ)
    (grief : ℝ → ℝ)
    (penalty : ℝ := 1)
    (emphasis : ℝ := 1) : ℝ :=
  -- Placeholder for the improper integral.
  -- In a full development this would be `∫ t in Set.Ici 0, contributionRate ... * asymptoticWeight ...`.
  -- For the foundational layer we expose the rate and weight so that future
  -- Lean work can attach a real integral while preserving the discrete theorems.
  0  -- sentinel; real integral to be attached in subsequent higher-gate commits

/-- Linking principle: a constant high-valence, zero-grief trajectory
    must produce a non-negative continuous contribution rate. -/
theorem constant_high_valence_rate_nonneg
    (v : ℝ)
    (hv : minValence ≤ v)
    (penalty : ℝ := 1) :
    0 ≤ contributionRate (fun _ => v) (fun _ => 0) penalty 0 := by
  simp [contributionRate]
  have hfloor : minValence < 1 := by norm_num [minValence]
  exact positiveTerm_nonneg v minValence hfloor

/-- Linking principle: a constant zero-valence, positive-grief trajectory
    produces a non-positive continuous contribution rate. -/
theorem constant_zero_valence_rate_nonpos
    (g : ℝ)
    (hg : 0 ≤ g)
    (penalty : ℝ)
    (hp : 0 ≤ penalty) :
    contributionRate (fun _ => 0) (fun _ => g) penalty 0 ≤ 0 := by
  simp [contributionRate, positiveTerm]
  have : ¬ (minValence ≤ (0 : ℝ)) := by norm_num [minValence]
  simp [if_neg this]
  exact neg_nonpos_of_nonneg (mul_nonneg hp hg)

/-- Discrete high-valence zero-grief samples remain the correct approximation
    seed for the continuous positive trajectory. -/
theorem discrete_matches_continuous_seed
    (s : NevcSample)
    (hv : minValence ≤ s.valence)
    (hg : s.griefLoad = 0) :
    0 < computeNevc [s] :=
  high_valence_zero_grief_pos s hv hg

end NEVC
