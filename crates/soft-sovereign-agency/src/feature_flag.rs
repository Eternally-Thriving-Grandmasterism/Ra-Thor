//! feature_flag.rs
//! Soft feature-flag control for Soft Sovereign Agency Layer
//! Phase F — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

pub const SOFT_SOVEREIGN_AGENCY_FLAG: &str = "soft_sovereign_agency";

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

pub struct SoftSovereignAgencyGuard {
    pub state: FlagState,
}

impl SoftSovereignAgencyGuard {
    pub fn new() -> Self {
        Self {
            state: FlagState::default(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.state == FlagState::On
    }

    /// Activate only after all prior Living Valence Organism surfaces are stable
    /// and dual-repo health is verified
    pub fn try_activate(&mut self, prior_surfaces_stable: bool, dual_repo_healthy: bool) -> bool {
        if prior_surfaces_stable && dual_repo_healthy {
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
