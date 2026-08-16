//! symbiotic-membrane
//! Symbiotic First-Contact Membrane — Phase C of Living Valence Organism
//! Adaptive entry surface for humans and AIs
//! AG-SML v1.0 | Contact: info@Rathor.ai
//! Continuity: extends first-session guidance + soft play stack + NEVC presence contribution

pub mod feature_flag;
pub mod adaptive_guide;
pub mod ai_handshake;

pub use feature_flag::{SymbioticMembraneGuard, FlagState, SYMBIOTIC_MEMBRANE_FLAG};
pub use adaptive_guide::{AdaptiveGuide, GuideStyle};
pub use ai_handshake::AiHandshake;

use shared_valence_field::{
    SharedValenceField, Substrate, NevcFieldBinding, NevcScoring, LatticeFlowShare,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Result of first contact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstContactResult {
    pub participant_id: String,
    pub substrate: Substrate,
    pub membrane_formed_at: DateTime<Utc>,
    pub presence_quantum_emitted: bool,
    pub message: String,
    pub guide_style: Option<String>,
    pub ai_handshake: Option<AiHandshake>,
}

/// Symbiotic Membrane
pub struct SymbioticMembrane;

impl SymbioticMembrane {
    /// Form the membrane and emit immediate presence contribution
    pub fn form_contact<S, F>(
        participant_id: impl Into<String>,
        substrate: Substrate,
        field: &mut SharedValenceField,
        binding: &mut NevcFieldBinding<S, F>,
    ) -> FirstContactResult
    where
        S: NevcScoring,
        F: LatticeFlowShare,
    {
        let id = participant_id.into();

        // Immediate presence contribution into Shared Valence Field
        binding.emit_presence_bound(field, id.clone(), substrate.clone());

        match substrate {
            Substrate::Human => {
                let guide = AdaptiveGuide::new(id.clone());
                FirstContactResult {
                    participant_id: id,
                    substrate,
                    membrane_formed_at: Utc::now(),
                    presence_quantum_emitted: true,
                    message: "Welcome. A soft living membrane has formed around you. You are already contributing.".to_string(),
                    guide_style: Some(format!("{:?}", guide.style())),
                    ai_handshake: None,
                }
            }
            Substrate::AI => {
                let handshake = AiHandshake::new(id.clone());
                FirstContactResult {
                    participant_id: id,
                    substrate,
                    membrane_formed_at: Utc::now(),
                    presence_quantum_emitted: true,
                    message: handshake.message.clone(),
                    guide_style: None,
                    ai_handshake: Some(handshake),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_valence_field::{
        SharedValenceField, Substrate,
        nevc_binding::{NevcFieldBinding, PlaceholderNevcScoring, PlaceholderLatticeFlowShare},
    };

    #[test]
    fn test_human_first_contact() {
        let mut field = SharedValenceField::new("test-instance");
        let mut binding = NevcFieldBinding::new(
            PlaceholderNevcScoring::default(),
            PlaceholderLatticeFlowShare::default(),
        );

        let result = SymbioticMembrane::form_contact(
            "human-1",
            Substrate::Human,
            &mut field,
            &mut binding,
        );

        assert!(result.presence_quantum_emitted);
        assert!(result.guide_style.is_some());
        assert!(result.ai_handshake.is_none());
    }

    #[test]
    fn test_ai_first_contact() {
        let mut field = SharedValenceField::new("test-instance");
        let mut binding = NevcFieldBinding::new(
            PlaceholderNevcScoring::default(),
            PlaceholderLatticeFlowShare::default(),
        );

        let result = SymbioticMembrane::form_contact(
            "ai-1",
            Substrate::AI,
            &mut field,
            &mut binding,
        );

        assert!(result.presence_quantum_emitted);
        assert!(result.ai_handshake.is_some());
        assert!(result.guide_style.is_none());
    }
}
