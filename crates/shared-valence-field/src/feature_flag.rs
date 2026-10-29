//! Flag defaults to off. `try_activate` stores `On` only when both bool arguments are true.
//! Contact: info@Rathor.ai

/// Canonical feature flag name (must match playbook)
pub const SHARED_VALENCE_FIELD_FLAG: &str = "shared_valence_field";

/// Simple runtime flag check (will later bind to the sealed SoftPolicyState / feature-flag system)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlagState {
    #[default]
    Off,
    On,
}

/// Flag wrapper. `is_active` is true only after `try_activate` stored `On`.
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

    /// Stores `On` only when both arguments are true. Does not measure valence.
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

impl Default for SharedValenceFieldGuard {
    fn default() -> Self {
        Self::new()
    }
}
