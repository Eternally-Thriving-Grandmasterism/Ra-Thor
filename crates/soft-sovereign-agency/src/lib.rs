//! soft-sovereign-agency
//! Soft Sovereign Agency Layer — Phase F of Living Valence Organism
//! Pure presentation and policy layer. No authority or persistence changes.
//! AG-SML v1.0 | Contact: info@Rathor.ai

use shared_valence_field::Substrate;
use serde::{Deserialize, Serialize};

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

/// Soft Sovereign Agency — presentation layer only
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

    /// Participant may switch views at any time (full sovereignty)
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.preferred_view = mode;
    }

    /// Returns the current preferred view
    pub fn current_view(&self) -> &ViewMode {
        &self.preferred_view
    }

    /// Guidance is always invitation — never coercion
    pub fn is_invitation_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_valence_field::Substrate;

    #[test]
    fn test_human_default_poetic() {
        let agency = SoftSovereignAgency::new("human-1", Substrate::Human);
        assert_eq!(agency.current_view(), &ViewMode::Poetic);
        assert!(agency.is_invitation_only());
    }

    #[test]
    fn test_ai_default_structured() {
        let agency = SoftSovereignAgency::new("ai-1", Substrate::AI);
        assert_eq!(agency.current_view(), &ViewMode::Structured);
    }

    #[test]
    fn test_fluid_switch() {
        let mut agency = SoftSovereignAgency::new("player-1", Substrate::Human);
        agency.set_view_mode(ViewMode::Structured);
        assert_eq!(agency.current_view(), &ViewMode::Structured);
    }
}
