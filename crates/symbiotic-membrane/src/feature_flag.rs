//! feature_flag.rs
//! Soft feature-flag control for Symbiotic Membrane
//! Phase C — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

pub const SYMBIOTIC_MEMBRANE_FLAG: &str = "symbiotic_membrane";

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

pub struct SymbioticMembraneGuard {
    pub state: FlagState,
}

impl SymbioticMembraneGuard {
    pub fn new() -> Self {
        Self {
            state: FlagState::default(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.state == FlagState::On
    }

    /// Activate only after Shared Valence Field is stable and dual-repo health is verified
    pub fn try_activate(&mut self, shared_valence_stable: bool, dual_repo_healthy: bool) -> bool {
        if shared_valence_stable && dual_repo_healthy {
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
