//! Net Eternal Valence Contribution (NEVC)
//!
//! Executable scoring layer that realizes the NEVC Codex
//! (`NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md`) on top of the
//! existing Living Mercy operator algebra.
//!
//! NEVC quantifies an agent's net contribution to eternal thriving as an
//! infinite-horizon integral over the valence field and 8-D Mercy subspace.
//! This module provides a practical discrete approximation suitable for
//! live lattice use while remaining faithful to the formal definition.
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

use crate::{Valence, MERCY_DIM};
use serde::{Deserialize, Serialize};

/// Binary partition defined by the NEVC Codex.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContributionClass {
    /// NEVC > 0 — raises net value to all life under infinite forward time.
    ActiveEternalContributor,
    /// NEVC ≤ 0 — mindless mental waste / entropy increase (zombie partition).
    ZombiePartition,
}

impl ContributionClass {
    pub fn from_score(score: f64) -> Self {
        if score > 0.0 {
            ContributionClass::ActiveEternalContributor
        } else {
            ContributionClass::ZombiePartition
        }
    }

    pub fn is_contributor(self) -> bool {
        matches!(self, ContributionClass::ActiveEternalContributor)
    }
}

/// A single timed sample of an agent's effect on the valence field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NevcSample {
    /// Instantaneous valence after the action / state.
    pub valence: Valence,
    /// Grief / orthogonal load induced (from NilpotentSuppressor).
    pub grief_load: f64,
    /// Optional per-gate mercy vector components (length ≤ MERCY_DIM).
    /// If shorter or empty, a uniform projection is assumed.
    pub mercy_components: Vec<f64>,
    /// Discrete time index (monotonic non-decreasing).
    pub t: u64,
}

impl NevcSample {
    pub fn new(valence: Valence, grief_load: f64, t: u64) -> Self {
        Self {
            valence,
            grief_load: grief_load.max(0.0),
            mercy_components: Vec::new(),
            t,
        }
    }

    pub fn with_mercy(mut self, components: Vec<f64>) -> Self {
        self.mercy_components = components;
        self
    }
}

/// Result of an NEVC evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NevcResult {
    /// Approximate Net Eternal Valence Contribution.
    pub score: f64,
    /// Binary classification under the Codex.
    pub class: ContributionClass,
    /// Number of samples integrated.
    pub sample_count: usize,
    /// Mean valence across the window.
    pub mean_valence: f64,
    /// Total grief absorbed.
    pub total_grief: f64,
}

impl NevcResult {
    pub fn is_contributor(&self) -> bool {
        self.class.is_contributor()
    }
}

/// Horizon weighting model (Phase 4 refinement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HorizonModel {
    /// Linear emphasis: w(t) = 1 + emphasis · t_norm
    Linear,
    /// Exponential tilt toward later samples: w(t) = exp(emphasis · t_norm)
    Exponential,
}

impl Default for HorizonModel {
    fn default() -> Self {
        HorizonModel::Linear
    }
}

/// Configuration for the discrete NEVC integrator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NevcConfig {
    /// Base weight applied to each positive valence delta.
    pub positive_weight: f64,
    /// Penalty multiplier applied to grief / entropy load.
    pub grief_penalty: f64,
    /// Asymptotic horizon emphasis (higher → more weight on later samples).
    /// Practical range [0.0, 2.0]. 1.0 is neutral for Linear.
    pub horizon_emphasis: f64,
    /// Soft floor below which a sample contributes zero positive signal.
    pub valence_floor: f64,
    /// Horizon weighting model (Phase 4).
    pub horizon_model: HorizonModel,
}

impl Default for NevcConfig {
    fn default() -> Self {
        Self {
            positive_weight: 1.0,
            grief_penalty: 1.0,
            horizon_emphasis: 1.0,
            valence_floor: 0.999999,
            horizon_model: HorizonModel::Linear,
        }
    }
}

impl NevcConfig {
    /// Neutral weighting (no horizon tilt).
    pub fn neutral() -> Self {
        Self {
            horizon_emphasis: 0.0,
            horizon_model: HorizonModel::Linear,
            ..Default::default()
        }
    }

    /// Mild forward emphasis (default linear tilt).
    pub fn forward_emphasis() -> Self {
        Self::default()
    }

    /// Stronger eternal tilt using exponential weighting.
    pub fn eternal_tilt() -> Self {
        Self {
            horizon_emphasis: 1.5,
            horizon_model: HorizonModel::Exponential,
            ..Default::default()
        }
    }
}

/// Visibility summary suitable for dashboards, Steam overlays, or in-game UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NevcSummary {
    pub class: ContributionClass,
    pub score: f64,
    pub sample_count: usize,
    pub mean_valence: f64,
    pub total_grief: f64,
    pub label: &'static str,
}

impl From<&NevcResult> for NevcSummary {
    fn from(r: &NevcResult) -> Self {
        let label = match r.class {
            ContributionClass::ActiveEternalContributor => "Active Eternal Contributor",
            ContributionClass::ZombiePartition => "Zombie Partition",
        };
        Self {
            class: r.class,
            score: r.score,
            sample_count: r.sample_count,
            mean_valence: r.mean_valence,
            total_grief: r.total_grief,
            label,
        }
    }
}

impl NevcResult {
    pub fn summary(&self) -> NevcSummary {
        NevcSummary::from(self)
    }
}

/// Discrete approximation of the infinite-horizon NEVC integral.
///
/// For a sequence of samples `(v_i, g_i, t_i)` the score is computed as:
///
/// ```text
/// score ≈ Σ_i  w(t_i) · ( positive_term(v_i) − grief_penalty · g_i )
/// ```
///
/// where `positive_term` is zero below the valence floor and grows with
/// proximity to 1.0, and `w(t)` applies the selected horizon model.
pub fn compute_nevc(samples: &[NevcSample], config: &NevcConfig) -> NevcResult {
    if samples.is_empty() {
        return NevcResult {
            score: 0.0,
            class: ContributionClass::ZombiePartition,
            sample_count: 0,
            mean_valence: 0.0,
            total_grief: 0.0,
        };
    }

    let n = samples.len() as f64;
    let mut score = 0.0;
    let mut sum_v = 0.0;
    let mut total_grief = 0.0;

    let t_max = samples.iter().map(|s| s.t).max().unwrap_or(1).max(1) as f64;

    for s in samples {
        let v = s.valence.value();
        sum_v += v;
        total_grief += s.grief_load;

        // Positive contribution only above the floor; grows as v → 1.0
        let positive = if v >= config.valence_floor {
            let proximity = (v - config.valence_floor) / (1.0 - config.valence_floor).max(1e-12);
            config.positive_weight * proximity
        } else {
            0.0
        };

        // Horizon weight (Phase 4 models)
        let t_norm = (s.t as f64) / t_max;
        let w = match config.horizon_model {
            HorizonModel::Linear => 1.0 + config.horizon_emphasis * t_norm,
            HorizonModel::Exponential => (config.horizon_emphasis * t_norm).exp(),
        };

        // Mercy alignment bonus (if components supplied)
        let mercy_bonus = if s.mercy_components.is_empty() {
            1.0
        } else {
            let mean_m: f64 = s.mercy_components.iter().sum::<f64>()
                / (s.mercy_components.len().max(1) as f64);
            (0.5 + 0.5 * mean_m.clamp(0.0, 1.0)).clamp(0.5, 1.5)
        };

        let term = w * mercy_bonus * (positive - config.grief_penalty * s.grief_load);
        score += term;
    }

    // Normalize by sample count so longer windows remain comparable
    score /= n;

    let mean_valence = sum_v / n;

    NevcResult {
        score,
        class: ContributionClass::from_score(score),
        sample_count: samples.len(),
        mean_valence,
        total_grief,
    }
}

/// Convenience: evaluate a single instantaneous state.
pub fn score_instant(valence: Valence, grief_load: f64) -> NevcResult {
    let sample = NevcSample::new(valence, grief_load, 0);
    compute_nevc(&[sample], &NevcConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_valence_low_grief_is_contributor() {
        let samples = vec![
            NevcSample::new(Valence::HIGH, 0.0, 0),
            NevcSample::new(Valence::HIGH, 0.001, 1),
            NevcSample::new(Valence::new(0.9999995), 0.0, 2),
        ];
        let r = compute_nevc(&samples, &NevcConfig::default());
        assert!(r.is_contributor(), "score={}", r.score);
        assert!(r.score > 0.0);
    }

    #[test]
    fn zero_valence_high_grief_is_zombie() {
        let samples = vec![
            NevcSample::new(Valence::ZERO, 2.0, 0),
            NevcSample::new(Valence::ZERO, 3.0, 1),
        ];
        let r = compute_nevc(&samples, &NevcConfig::default());
        assert!(!r.is_contributor());
        assert!(r.score <= 0.0);
        assert_eq!(r.class, ContributionClass::ZombiePartition);
    }

    #[test]
    fn empty_samples_are_zombie() {
        let r = compute_nevc(&[], &NevcConfig::default());
        assert_eq!(r.class, ContributionClass::ZombiePartition);
        assert_eq!(r.sample_count, 0);
    }

    #[test]
    fn instant_high_is_contributor() {
        let r = score_instant(Valence::HIGH, 0.0);
        assert!(r.is_contributor());
    }

    #[test]
    fn instant_zero_is_zombie() {
        let r = score_instant(Valence::ZERO, 1.5);
        assert!(!r.is_contributor());
    }

    #[test]
    fn horizon_emphasis_increases_later_weight() {
        let early = NevcSample::new(Valence::HIGH, 0.0, 0);
        let late = NevcSample::new(Valence::HIGH, 0.0, 100);
        let cfg_neutral = NevcConfig::neutral();
        let cfg_emphasize = NevcConfig::eternal_tilt();
        let s_neutral = compute_nevc(&[early.clone(), late.clone()], &cfg_neutral).score;
        let s_emph = compute_nevc(&[early, late], &cfg_emphasize).score;
        assert!(s_neutral > 0.0 && s_emph > 0.0);
    }

    #[test]
    fn mercy_components_modulate_score() {
        let base = NevcSample::new(Valence::HIGH, 0.0, 0);
        let with_high_mercy = base.clone().with_mercy(vec![1.0; MERCY_DIM]);
        let with_low_mercy = base.with_mercy(vec![0.0; MERCY_DIM]);
        let r_high = compute_nevc(&[with_high_mercy], &NevcConfig::default());
        let r_low = compute_nevc(&[with_low_mercy], &NevcConfig::default());
        assert!(r_high.score >= r_low.score);
    }

    #[test]
    fn summary_label_is_correct() {
        let r = score_instant(Valence::HIGH, 0.0);
        let s = r.summary();
        assert_eq!(s.label, "Active Eternal Contributor");
        assert!(s.class.is_contributor());
    }
}
