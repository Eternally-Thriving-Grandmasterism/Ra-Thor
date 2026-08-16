//! feature_flag.rs
//! Soft feature-flag control for Abundance Breath Loop
//! Phase E — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

pub const ABUNDANCE_BREATH_FLAG: &str = "abundance_breath";

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

pub struct AbundanceBreathGuard {
    pub state: FlagState,
}

impl AbundanceBreathGuard {
    pub fn new() -> Self {
        Self {
            state: FlagState::default(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.state == FlagState::On
    }

    /// Activate only after Epiphany Bridge is stable and dual-repo health is verified
    pub fn try_activate(&mut self, epiphany_stable: bool, dual_repo_healthy: bool) -> bool {
        if epiphany_stable && dual_repo_healthy {
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
