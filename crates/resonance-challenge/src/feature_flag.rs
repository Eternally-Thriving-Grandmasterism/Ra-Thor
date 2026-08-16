//! feature_flag.rs
//! Soft feature-flag control for Resonance Challenge Conductor
//! Phase D — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

pub const RESONANCE_CHALLENGE_FLAG: &str = "resonance_challenge";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagState {
    Off,
    On,
}

impl Default for FlagState {
    fn default() -> Self {
        FlagState::Off
    }
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

    /// Activate only after Symbiotic Membrane is stable and dual-repo health is verified
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
