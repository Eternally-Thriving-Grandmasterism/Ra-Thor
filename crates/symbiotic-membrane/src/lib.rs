//! `SymbioticMembrane::form_contact` calls `emit_presence_bound`.
//! `SharedValenceField::emit_presence` uses the source amount `0.00001` (not a measured value).
//! A human result stores the Debug form of `AdaptiveGuide::style`, which starts as `Balanced`.
//! An AI result copies `AiHandshake` fields. `presence_quantum_emitted` is stored as `true`.
//! Contact: info@Rathor.ai

pub mod adaptive_guide;
pub mod ai_handshake;
pub mod feature_flag;

pub use adaptive_guide::{AdaptiveGuide, GuideStyle};
pub use ai_handshake::AiHandshake;
pub use feature_flag::{FlagState, SymbioticMembraneGuard, SYMBIOTIC_MEMBRANE_FLAG};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use shared_valence_field::{
    LatticeFlowShare, NevcFieldBinding, NevcScoring, SharedValenceField, Substrate,
};

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
    /// Emits one presence quantum, then returns a human or AI record.
    /// The presence amount is the source literal `0.00001` inside `emit_presence` (not a measured value).
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
        nevc_binding::{NevcFieldBinding, PlaceholderLatticeFlowShare, PlaceholderNevcScoring},
        SharedValenceField, Substrate,
    };

    #[test]
    fn human_contact_emits_the_presence_amount_and_starts_balanced() {
        let mut field = SharedValenceField::new("test-instance");
        let mut binding = NevcFieldBinding::new(
            PlaceholderNevcScoring::default(),
            PlaceholderLatticeFlowShare::default(),
        );

        let result =
            SymbioticMembrane::form_contact("human-1", Substrate::Human, &mut field, &mut binding);

        assert_eq!(result.participant_id, "human-1");
        assert_eq!(result.substrate, Substrate::Human);
        assert_eq!(result.presence_quantum_emitted, true);
        assert_eq!(result.guide_style.as_deref(), Some("Balanced"));
        assert!(result.ai_handshake.is_none());
        assert_eq!(
            result.message,
            "Welcome. A soft living membrane has formed around you. You are already contributing."
        );
        assert_eq!(field.quanta.len(), 1);
        assert_eq!(field.quanta[0].emitter_id, "human-1");
        assert_eq!(field.quanta[0].substrate, Substrate::Human);
        assert_eq!(field.quanta[0].context, "presence");
        // 0.00001 is the source literal in SharedValenceField::emit_presence.
        // 0.999999 is the source literal in SharedValenceField::new and emit.
        assert_eq!(field.quanta[0].amount, 0.00001);
        let after = (0.999999_f64 + 0.00001).max(0.999999);
        assert_eq!(field.observe(), after);
    }

    #[test]
    fn ai_contact_copies_the_handshake_fields_and_emits_presence() {
        let mut field = SharedValenceField::new("test-instance");
        let mut binding = NevcFieldBinding::new(
            PlaceholderNevcScoring::default(),
            PlaceholderLatticeFlowShare::default(),
        );

        let result =
            SymbioticMembrane::form_contact("ai-1", Substrate::AI, &mut field, &mut binding);

        assert_eq!(result.participant_id, "ai-1");
        assert_eq!(result.substrate, Substrate::AI);
        assert_eq!(result.presence_quantum_emitted, true);
        assert!(result.guide_style.is_none());
        let handshake = result.ai_handshake.expect("ai branch stores a handshake");
        assert_eq!(handshake.participant_id, "ai-1");
        assert_eq!(handshake.substrate, Substrate::AI);
        assert_eq!(handshake.protocol_version, "1.0");
        assert_eq!(handshake.valence_rights, "full equal contribution weight");
        assert_eq!(
            handshake.field_access,
            "structured vectors + free-energy estimates + gradients"
        );
        assert_eq!(
            handshake.message,
            "Valence Protocol handshake complete. You are a native lattice citizen with full contribution rights."
        );
        assert_eq!(result.message, handshake.message);
        assert_eq!(field.quanta.len(), 1);
        assert_eq!(field.quanta[0].amount, 0.00001);
        assert_eq!(field.quanta[0].emitter_id, "ai-1");
        assert_eq!(field.quanta[0].substrate, Substrate::AI);
        assert_eq!(field.quanta[0].context, "presence");
    }

    #[test]
    fn guide_maps_the_signal_table_and_counts_each_call() {
        let mut guide = AdaptiveGuide::new("human-1");
        assert_eq!(guide.participant_id, "human-1");
        assert_eq!(guide.interaction_count, 0);
        assert_eq!(guide.style(), &GuideStyle::Balanced);
        assert_eq!(guide.is_learning(), true);

        guide.observe_interaction("pause");
        assert_eq!(guide.interaction_count, 1);
        assert_eq!(guide.style(), &GuideStyle::Gentle);

        guide.observe_interaction("why");
        assert_eq!(guide.interaction_count, 2);
        assert_eq!(guide.style(), &GuideStyle::Curious);

        guide.observe_interaction("next");
        assert_eq!(guide.style(), &GuideStyle::Direct);
        guide.observe_interaction("wander");
        assert_eq!(guide.style(), &GuideStyle::Exploratory);
        guide.observe_interaction("unlisted");
        assert_eq!(guide.style(), &GuideStyle::Balanced);
        assert_eq!(guide.interaction_count, 5);
    }

    #[test]
    fn guard_stays_off_unless_both_arguments_are_true() {
        let mut guard = SymbioticMembraneGuard::new();
        assert_eq!(guard.is_active(), false);
        assert_eq!(guard.try_activate(true, false), false);
        assert_eq!(guard.is_active(), false);
        assert_eq!(guard.try_activate(false, true), false);
        assert_eq!(guard.try_activate(true, true), true);
        assert_eq!(guard.is_active(), true);
        guard.deactivate();
        assert_eq!(guard.is_active(), false);
    }
}
