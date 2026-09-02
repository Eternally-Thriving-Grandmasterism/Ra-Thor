-- lean/NEVC.lean
-- Net Eternal Valence Contribution (NEVC) Formalization
-- Phase 9 + Finish Pass D: continuous strengthening + improper-limit sketch

/-!
# Net Eternal Valence Contribution (NEVC)

Formalization of the NEVC Codex.
Finish Pass D adds an improper-integral limit sketch (Tendsto-style Props)
without requiring the full Mathlib analysis surface in-tree.

AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
Thunder locked in. Yoi ⚡
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Basic
import Mathlib.Algebra.Order.Field.Basic

import TOLC8_MercyGate

namespace NEVC

open TOLC

/-! ## Binary Partition (Codex §3.4) -/

inductive ContributionClass where
  | ActiveEternalContributor
  | ZombiePartition
deriving DecidableEq, Repr

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

/-! ## Discrete Sample -/

structure NevcSample where
  valence    : ℝ
  griefLoad  : ℝ
  t          : ℕ
  valence_nonneg : 0 ≤ valence
  valence_le_one : valence ≤ 1
  grief_nonneg   : 0 ≤ griefLoad

def positiveTerm (v : ℝ) (floor : ℝ := minValence) : ℝ :=
  if floor ≤ v then (v - floor) / (1 - floor) else 0

theorem positiveTerm_nonneg (v floor : ℝ) (hfloor : floor < 1) :
    0 ≤ positiveTerm v floor := by
  simp [positiveTerm]
  split_ifs with h
  · apply div_nonneg <;> linarith
  · exact le_refl 0

theorem positiveTerm_zero_below (v floor : ℝ) (h : v < floor) :
    positiveTerm v floor = 0 := by
  simp [positiveTerm, not_le.mpr h]

def computeNevc (samples : List NevcSample) (griefPenalty : ℝ := 1) : ℝ :=
  match samples with
  | [] => 0
  | _ =>
    let n : ℝ := samples.length
    let raw := samples.foldl (fun acc s =>
      acc + (positiveTerm s.valence - griefPenalty * s.griefLoad)) 0
    raw / n

/-! ## Discrete Core Properties -/

theorem empty_is_zombie :
    classFromScore (computeNevc []) = ContributionClass.ZombiePartition := by
  simp [computeNevc, classFromScore]

theorem high_valence_zero_grief_pos
    (s : NevcSample) (hv : minValence ≤ s.valence) (hg : s.griefLoad = 0) :
    0 < computeNevc [s] := by
  simp [computeNevc, positiveTerm, hg]
  have hfloor : minValence < 1 := by norm_num [minValence]
  have hdiv : 0 < (s.valence - minValence) / (1 - minValence) := by
    apply div_pos <;> linarith
  simpa [if_pos hv] using hdiv

theorem high_valence_zero_grief_contributor
    (s : NevcSample) (hv : minValence ≤ s.valence) (hg : s.griefLoad = 0) :
    classFromScore (computeNevc [s]) = ContributionClass.ActiveEternalContributor := by
  apply classFromScore_pos
  exact high_valence_zero_grief_pos s hv hg

theorem zero_valence_pos_grief_nonpos
    (s : NevcSample) (hv : s.valence = 0) (hg : 0 < s.griefLoad) :
    computeNevc [s] (griefPenalty := 1) ≤ 0 := by
  simp [computeNevc, positiveTerm, hv]
  have : ¬ (minValence ≤ 0) := by norm_num [minValence]
  simp [if_neg this]
  linarith

theorem zero_valence_pos_grief_zombie
    (s : NevcSample) (hv : s.valence = 0) (hg : 0 < s.griefLoad) :
    classFromScore (computeNevc [s]) = ContributionClass.ZombiePartition := by
  apply classFromScore_nonpos
  exact zero_valence_pos_grief_nonpos s hv hg

theorem pure_high_valence_pos
    (samples : List NevcSample) (hne : samples ≠ [])
    (hall : ∀ s ∈ samples, minValence ≤ s.valence ∧ s.griefLoad = 0) :
    0 < computeNevc samples := by
  cases samples with
  | nil => contradiction
  | cons s rest =>
    have hs := hall s (List.mem_cons_self s rest)
    simp [computeNevc]
    have : 0 < positiveTerm s.valence := by
      simpa [positiveTerm, hs.2, if_pos hs.1] using
        (div_pos (by linarith : 0 < s.valence - minValence) (by linarith : 0 < 1 - minValence))
    apply div_pos
    · have fold_ge : (s :: rest).foldl (fun acc t => acc + (positiveTerm t.valence - 1 * t.griefLoad)) 0
                   ≥ positiveTerm s.valence := by
        simp [hs.2]; exact le_refl _
      linarith [this]
    · exact Nat.cast_pos.mpr (List.length_pos_of_ne_nil hne)

theorem below_floor_nonpos_term
    (s : NevcSample) (hbelow : s.valence < minValence) :
    positiveTerm s.valence - 1 * s.griefLoad ≤ 0 := by
  have hpt : positiveTerm s.valence = 0 := positiveTerm_zero_below s.valence minValence hbelow
  simp [hpt]
  exact neg_nonpos_of_nonneg s.grief_nonneg

/-! ## Continuous Strengthening -/

def contributionRate (valence : ℝ → ℝ) (grief : ℝ → ℝ) (penalty : ℝ) (t : ℝ) : ℝ :=
  positiveTerm (valence t) - penalty * (grief t)

def asymptoticWeight (emphasis : ℝ) (t : ℝ) : ℝ :=
  1 + emphasis * t

theorem asymptoticWeight_pos (emphasis t : ℝ) (he : 0 ≤ emphasis) (ht : 0 ≤ t) :
    0 < asymptoticWeight emphasis t := by
  simp [asymptoticWeight]; linarith

theorem asymptoticWeight_mono (emphasis : ℝ) (he : 0 ≤ emphasis)
    (t1 t2 : ℝ) (h : t1 ≤ t2) :
    asymptoticWeight emphasis t1 ≤ asymptoticWeight emphasis t2 := by
  simp [asymptoticWeight]; nlinarith

def integrand (valence : ℝ → ℝ) (grief : ℝ → ℝ)
    (penalty emphasis : ℝ) (t : ℝ) : ℝ :=
  contributionRate valence grief penalty t * asymptoticWeight emphasis t

def IntegrableOnCompact
    (valence : ℝ → ℝ) (grief : ℝ → ℝ) (penalty emphasis T : ℝ) : Prop :=
  ∃ M : ℝ, ∀ t : ℝ, 0 ≤ t → t ≤ T → |integrand valence grief penalty emphasis t| ≤ M

def ImproperIntegrable
    (valence : ℝ → ℝ) (grief : ℝ → ℝ) (penalty emphasis : ℝ) : Prop :=
  ∃ B : ℝ, ∀ T : ℝ, 0 ≤ T →
    IntegrableOnCompact valence grief penalty emphasis T

def finiteHorizonApprox
    (valence : ℝ → ℝ) (grief : ℝ → ℝ)
    (penalty emphasis T : ℝ) (n : ℕ) : ℝ :=
  if n = 0 ∨ T ≤ 0 then 0
  else
    let dt := T / (n : ℝ)
    let rec sum : ℕ → ℝ
      | 0 => 0
      | k + 1 =>
        let t := (k : ℝ) * dt
        sum k + integrand valence grief penalty emphasis t * dt
    sum n

def continuousNevcApprox
    (valence : ℝ → ℝ) (grief : ℝ → ℝ)
    (penalty emphasis T : ℝ) (n : ℕ) : ℝ :=
  finiteHorizonApprox valence grief penalty emphasis T n

def continuousNevc
    (valence : ℝ → ℝ) (grief : ℝ → ℝ)
    (penalty : ℝ := 1) (emphasis : ℝ := 1)
    (T : ℝ := 1) (n : ℕ := 10) : ℝ :=
  continuousNevcApprox valence grief penalty emphasis T n

/-! ## Finish Pass D — Improper limit (Tendsto-style sketch) -/

/-- ε-δ style limit statement for the horizon family T ↦ I(T).
    Stands in for `Tendsto I atTop (𝓝 L)` without requiring full Topology imports. -/
def HasImproperLimit (I : ℝ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, 0 < ε → ∃ T0 : ℝ, ∀ T : ℝ, T0 ≤ T → |I T - L| < ε

/-- The continuous NEVC ideal is the improper limit of finite-horizon values
    when such a limit exists. -/
def continuousNevcLimit
    (valence : ℝ → ℝ) (grief : ℝ → ℝ)
    (penalty emphasis : ℝ) (n : ℕ) (L : ℝ) : Prop :=
  HasImproperLimit (fun T => continuousNevcApprox valence grief penalty emphasis T n) L

/-- For a constant high-valence zero-grief trajectory the integrand is nonnegative,
    so partial integrals are monotone in T when the weight is nonnegative. -/
theorem constant_high_valence_integrand_nonneg
    (v : ℝ) (hv : minValence ≤ v)
    (penalty emphasis t : ℝ)
    (he : 0 ≤ emphasis) (ht : 0 ≤ t) :
    0 ≤ integrand (fun _ => v) (fun _ => 0) penalty emphasis t := by
  simp [integrand, contributionRate]
  have hfloor : minValence < 1 := by norm_num [minValence]
  have hpt : 0 ≤ positiveTerm v := positiveTerm_nonneg v minValence hfloor
  have hw : 0 ≤ asymptoticWeight emphasis t := le_of_lt (asymptoticWeight_pos emphasis t he ht)
  exact mul_nonneg hpt hw

/-- Linking: if the improper limit exists and is strictly positive, the continuous
    object classifies as ActiveEternalContributor. -/
theorem improper_limit_pos_is_contributor
    (L : ℝ) (hL : 0 < L) :
    classFromScore L = ContributionClass.ActiveEternalContributor :=
  classFromScore_pos L hL

/-- Linking: non-positive improper limit classifies as ZombiePartition. -/
theorem improper_limit_nonpos_is_zombie
    (L : ℝ) (hL : L ≤ 0) :
    classFromScore L = ContributionClass.ZombiePartition :=
  classFromScore_nonpos L hL

theorem constant_high_valence_rate_nonneg
    (v : ℝ) (hv : minValence ≤ v) (penalty : ℝ := 1) :
    0 ≤ contributionRate (fun _ => v) (fun _ => 0) penalty 0 := by
  simp [contributionRate]
  have hfloor : minValence < 1 := by norm_num [minValence]
  exact positiveTerm_nonneg v minValence hfloor

theorem constant_zero_valence_rate_nonpos
    (g : ℝ) (hg : 0 ≤ g) (penalty : ℝ) (hp : 0 ≤ penalty) :
    contributionRate (fun _ => 0) (fun _ => g) penalty 0 ≤ 0 := by
  simp [contributionRate, positiveTerm]
  have : ¬ (minValence ≤ (0 : ℝ)) := by norm_num [minValence]
  simp [if_neg this]
  exact neg_nonpos_of_nonneg (mul_nonneg hp hg)

theorem discrete_matches_continuous_seed
    (s : NevcSample) (hv : minValence ≤ s.valence) (hg : s.griefLoad = 0) :
    0 < computeNevc [s] :=
  high_valence_zero_grief_pos s hv hg

def classFromContinuousApprox
    (valence : ℝ → ℝ) (grief : ℝ → ℝ)
    (penalty emphasis T : ℝ) (n : ℕ) : ContributionClass :=
  classFromScore (continuousNevcApprox valence grief penalty emphasis T n)

end NEVC
