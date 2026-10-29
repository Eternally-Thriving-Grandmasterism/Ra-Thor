//! feature_flag.rs
//! Soft feature-flag control for Resonance Challenge Conductor
//! Phase D — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

pub const RESONANCE_CHALLENGE_FLAG: &str = "resonance_challenge";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlagState {
    #[default]
    Off,
    On,
}

pub struct ResonanceChallengeGuard {
    pub state: FlagState,
}

impl ResonanceChallengeGuard {
    pub fn new() -> Self {
        Self {
            state: FlagState::default(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.state == FlagState::On
    }

    /// Stores `On` only when both arguments are true. Does not measure valence or health.
    pub fn try_activate(&mut self, membrane_stable: bool, dual_repo_healthy: bool) -> bool {
        if membrane_stable && dual_repo_healthy {
            self.state = FlagState::On;
            true
        } else {
            self.state = FlagState::Off;
            false
        }
    }

    pub fn deactivate(&mut self) {
        self.state = FlagState::Off;
    }
}

impl Default for ResonanceChallengeGuard {
    fn default() -> Self {
        Self::new()
    }
}
