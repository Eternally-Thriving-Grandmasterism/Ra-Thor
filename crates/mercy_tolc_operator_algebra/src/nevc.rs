//! Net Eternal Valence Contribution (NEVC)
//!
//! Executable scoring layer that realizes the NEVC Codex
//! (`NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md`) on top of the
//! existing Living Mercy operator algebra.
//!
//! Finish Pass D: Compassion recovery state (transient trauma is not sealed).
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

/// Finish Pass D — Compassion-gate recovery policy (Codex §6).
///
/// Transient low-valence / trauma-linked states must not be permanently sealed
/// as zombie without recovery trajectory evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompassionRecoveryState {
    /// Recovery pathways remain open (default for borderline / trauma cases).
    Open,
    /// Classification is durable for this window (sustained pattern, not transient).
    Sealed,
}

impl Default for CompassionRecoveryState {
    fn default() -> Self {
        CompassionRecoveryState::Open
    }
}

impl CompassionRecoveryState {
    pub fn is_open(self) -> bool {
        matches!(self, CompassionRecoveryState::Open)
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
    pub mercy_components: Vec<f64>,
    /// Discrete time index (monotonic non-decreasing).
    pub t: u64,
    /// Optional flag: this sample is trauma-linked / transient (keeps recovery Open).
    #[serde(default)]
    pub transient: bool,
}

impl NevcSample {
    pub fn new(valence: Valence, grief_load: f64, t: u64) -> Self {
        Self {
            valence,
            grief_load: grief_load.max(0.0),
            mercy_components: Vec::new(),
            t,
            transient: false,
        }
    }

    pub fn with_mercy(mut self, components: Vec<f64>) -> Self {
        self.mercy_components = components;
        self
    }

    pub fn transient(mut self) -> Self {
        self.transient = true;
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
    /// Finish Pass D: whether Compassion recovery remains open.
    pub recovery: CompassionRecoveryState,
}

impl NevcResult {
    pub fn is_contributor(&self) -> bool {
        self.class.is_contributor()
    }

    pub fn recovery_open(&self) -> bool {
        self.recovery.is_open()
    }
}

/// Horizon weighting model (Phase 4 refinement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HorizonModel {
    Linear,
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
    pub positive_weight: f64,
    pub grief_penalty: f64,
    pub horizon_emphasis: f64,
    pub valence_floor: f64,
    pub horizon_model: HorizonModel,
    /// If any sample is marked transient, keep recovery Open even when score ≤ 0.
    pub respect_transient: bool,
}

impl Default for NevcConfig {
    fn default() -> Self {
        Self {
            positive_weight: 1.0,
            grief_penalty: 1.0,
            horizon_emphasis: 1.0,
            valence_floor: 0.999999,
            horizon_model: HorizonModel::Linear,
            respect_transient: true,
        }
    }
}

impl NevcConfig {
    pub fn neutral() -> Self {
        Self {
            horizon_emphasis: 0.0,
            horizon_model: HorizonModel::Linear,
            ..Default::default()
        }
    }

    pub fn forward_emphasis() -> Self {
        Self::default()
    }

    pub fn eternal_tilt() -> Self {
        Self {
            horizon_emphasis: 1.5,
            horizon_model: HorizonModel::Exponential,
            ..Default::default()
        }
    }
}

/// Visibility summary suitable for dashboards / overlays.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NevcSummary {
    pub class: ContributionClass,
    pub score: f64,
    pub sample_count: usize,
    pub mean_valence: f64,
    pub total_grief: f64,
    pub label: &'static str,
    pub recovery: CompassionRecoveryState,
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
            recovery: r.recovery,
        }
    }
}

impl NevcResult {
    pub fn summary(&self) -> NevcSummary {
        NevcSummary::from(self)
    }
}

/// Discrete approximation of the infinite-horizon NEVC integral.
pub fn compute_nevc(samples: &[NevcSample], config: &NevcConfig) -> NevcResult {
    if samples.is_empty() {
        return NevcResult {
            score: 0.0,
            class: ContributionClass::ZombiePartition,
            sample_count: 0,
            mean_valence: 0.0,
            total_grief: 0.0,
            // Empty window: recovery remains open (no sustained pattern yet).
            recovery: CompassionRecoveryState::Open,
        };
    }

    let n = samples.len() as f64;
    let mut score = 0.0;
    let mut sum_v = 0.0;
    let mut total_grief = 0.0;
    let mut any_transient = false;

    let t_max = samples.iter().map(|s| s.t).max().unwrap_or(1).max(1) as f64;

    for s in samples {
        any_transient |= s.transient;
        let v = s.valence.value();
        sum_v += v;
        total_grief += s.grief_load;

        let positive = if v >= config.valence_floor {
            let proximity = (v - config.valence_floor) / (1.0 - config.valence_floor).max(1e-12);
            config.positive_weight * proximity
        } else {
            0.0
        };

        let t_norm = (s.t as f64) / t_max;
        let w = match config.horizon_model {
            HorizonModel::Linear => 1.0 + config.horizon_emphasis * t_norm,
            HorizonModel::Exponential => (config.horizon_emphasis * t_norm).exp(),
        };

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

    score /= n;
    let mean_valence = sum_v / n;
    let class = ContributionClass::from_score(score);

    // Compassion policy: transient samples keep recovery Open; sustained
    // negative pattern may Seal. Contributors always leave recovery Open.
    let recovery = if class.is_contributor() {
        CompassionRecoveryState::Open
    } else if config.respect_transient && any_transient {
        CompassionRecoveryState::Open
    } else if samples.len() >= 3 && score < -0.5 {
        // Sustained strong negative window → seal for this evaluation only.
        CompassionRecoveryState::Sealed
    } else {
        CompassionRecoveryState::Open
    };

    NevcResult {
        score,
        class,
        sample_count: samples.len(),
        mean_valence,
        total_grief,
        recovery,
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
        assert!(r.recovery_open());
    }

    #[test]
    fn zero_valence_high_grief_is_zombie() {
        let samples = vec![
            NevcSample::new(Valence::ZERO, 2.0, 0),
            NevcSample::new(Valence::ZERO, 3.0, 1),
        ];
        let r = compute_nevc(&samples, &NevcConfig::default());
        assert!(!r.is_contributor());
        assert_eq!(r.class, ContributionClass::ZombiePartition);
    }

    #[test]
    fn empty_samples_are_zombie_with_open_recovery() {
        let r = compute_nevc(&[], &NevcConfig::default());
        assert_eq!(r.class, ContributionClass::ZombiePartition);
        assert!(r.recovery_open());
    }

    #[test]
    fn transient_keeps_recovery_open() {
        let samples = vec![
            NevcSample::new(Valence::ZERO, 2.0, 0).transient(),
            NevcSample::new(Valence::ZERO, 2.0, 1).transient(),
            NevcSample::new(Valence::ZERO, 2.0, 2).transient(),
        ];
        let r = compute_nevc(&samples, &NevcConfig::default());
        assert!(!r.is_contributor());
        assert!(r.recovery_open(), "transient trauma must not seal");
    }

    #[test]
    fn instant_high_is_contributor() {
        let r = score_instant(Valence::HIGH, 0.0);
        assert!(r.is_contributor());
    }

    #[test]
    fn summary_includes_recovery() {
        let r = score_instant(Valence::HIGH, 0.0);
        let s = r.summary();
        assert_eq!(s.label, "Active Eternal Contributor");
        assert_eq!(s.recovery, CompassionRecoveryState::Open);
    }
}
