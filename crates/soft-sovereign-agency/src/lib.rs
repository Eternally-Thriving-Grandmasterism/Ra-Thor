//! `SoftSovereignAgency` stores a participant id, a substrate, and a view mode.
//! `new` sets Human to `Poetic` and AI to `Structured`. `set_view_mode` replaces the mode.
//! `is_invitation_only` returns true. This crate does not change authority or persistence.
//! `src/feature_flag.rs` and `src/dual_path.rs` are not modules.
//! `src/dual_path.rs` contains the source literal `0.999999` in a string. That file is not compiled.
//! Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};
use shared_valence_field::Substrate;

/// Preferred view mode for a participant
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViewMode {
    /// Human poetic / sensory expression
    Poetic,
    /// AI structured / mathematical expression
    Structured,
    /// Fluid switching allowed at any moment
    Fluid,
}

/// Stored view preference. Changing it does not write the field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftSovereignAgency {
    pub participant_id: String,
    pub substrate: Substrate,
    pub preferred_view: ViewMode,
}

impl SoftSovereignAgency {
    pub fn new(participant_id: impl Into<String>, substrate: Substrate) -> Self {
        let preferred_view = match substrate {
            Substrate::Human => ViewMode::Poetic,
            Substrate::AI => ViewMode::Structured,
        };

        Self {
            participant_id: participant_id.into(),
            substrate,
            preferred_view,
        }
    }

    /// Replaces `preferred_view`. No other field changes.
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.preferred_view = mode;
    }

    /// Returns the current preferred view
    pub fn current_view(&self) -> &ViewMode {
        &self.preferred_view
    }

    /// Returns true. The body does not read `self`.
    pub fn is_invitation_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_valence_field::Substrate;

    #[test]
    fn human_starts_poetic_and_keeps_the_participant_id() {
        let agency = SoftSovereignAgency::new("human-1", Substrate::Human);
        assert_eq!(agency.participant_id, "human-1");
        assert_eq!(agency.substrate, Substrate::Human);
        assert_eq!(agency.current_view(), &ViewMode::Poetic);
        assert_eq!(agency.preferred_view, ViewMode::Poetic);
    }

    #[test]
    fn ai_starts_structured() {
        let agency = SoftSovereignAgency::new("ai-1", Substrate::AI);
        assert_eq!(agency.participant_id, "ai-1");
        assert_eq!(agency.substrate, Substrate::AI);
        assert_eq!(agency.current_view(), &ViewMode::Structured);
    }

    #[test]
    fn set_view_mode_replaces_poetic_with_structured_then_fluid() {
        let mut agency = SoftSovereignAgency::new("player-1", Substrate::Human);
        assert_eq!(agency.current_view(), &ViewMode::Poetic);
        agency.set_view_mode(ViewMode::Structured);
        assert_eq!(agency.participant_id, "player-1");
        assert_eq!(agency.substrate, Substrate::Human);
        assert_eq!(agency.current_view(), &ViewMode::Structured);
        agency.set_view_mode(ViewMode::Fluid);
        assert_eq!(agency.current_view(), &ViewMode::Fluid);
        assert_eq!(agency.preferred_view, ViewMode::Fluid);
    }
}
