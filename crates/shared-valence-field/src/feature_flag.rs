//! feature_flag.rs
//! Soft feature-flag control for Shared Valence Field
//! Phase B — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

/// Canonical feature flag name (must match playbook)
pub const SHARED_VALENCE_FIELD_FLAG: &str = "shared_valence_field";

/// Simple runtime flag check (will later bind to the sealed SoftPolicyState / feature-flag system)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagState {
    Off,
    On,
}

impl Default for FlagState {
    fn default() -> Self {
        FlagState::Off // default off per playbook
    }
}

/// Guard that ensures the Shared Valence Field only activates when the flag is On
/// and all success criteria (valence floor, dual-repo health) are met
pub struct SharedValenceFieldGuard {
    pub state: FlagState,
}

impl SharedValenceFieldGuard {
    pub fn new() -> Self {
        Self {
            state: FlagState::default(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.state == FlagState::On
    }

    /// Activate only after success criteria are verified (valence metrics + dual-repo health)
    pub fn try_activate(&mut self, valence_ok: bool, dual_repo_healthy: bool) -> bool {
        if valence_ok && dual_repo_healthy {
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
