//! crates/powrush/src/npc/dialogue.rs
//! Production-grade Dialogue Selection System for Powrush
//! Context-aware, mercy-gated, relationship-driven responses | v1.0 | AG-SML v1.0

use super::{NpcBlackboard, Relationship, RelationshipLevel};

#[derive(Debug, Clone)]
pub struct DialogueResponse {
    pub text: String,
    pub tone: DialogueTone,
    pub mercy_impact: f64,
    pub requires_relationship: Option<RelationshipLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogueTone {
    Hostile,
    Wary,
    Neutral,
    Friendly,
    Reverent,
    Joyful,
}

pub struct DialogueSystem;

impl DialogueSystem {
    /// Selects the most appropriate dialogue response based on full context
    pub fn select_response(
        blackboard: &NpcBlackboard,
        relationship: &Relationship,
        situation: &str,
    ) -> DialogueResponse {
        let level = relationship.level;

        // Hostile path
        if level == RelationshipLevel::Hostile || blackboard.current_mercy_valence < 0.3 {
            return DialogueResponse {
                text: "Stay back. I don't trust you.".to_string(),
                tone: DialogueTone::Hostile,
                mercy_impact: -0.15,
                requires_relationship: Some(RelationshipLevel::Hostile),
            };
        }

        // High mercy + high relationship = warm responses
        if blackboard.world_mercy > 0.8 && relationship.is_friendly() {
            if situation.contains("help") || situation.contains("need") {
                return DialogueResponse {
                    text: "Of course. How can I help you on your path?".to_string(),
                    tone: DialogueTone::Joyful,
                    mercy_impact: 0.25,
                    requires_relationship: Some(RelationshipLevel::Friendly),
                };
            }
        }

        // Default context-aware responses
        match level {
            RelationshipLevel::Devoted | RelationshipLevel::Revered => DialogueResponse {
                text: "It is an honor to speak with you. Your presence brings light.".to_string(),
                tone: DialogueTone::Reverent,
                mercy_impact: 0.1,
                requires_relationship: None,
            },
            RelationshipLevel::Friendly => DialogueResponse {
                text: "Good to see you again. Things feel better when you're around.".to_string(),
                tone: DialogueTone::Friendly,
                mercy_impact: 0.08,
                requires_relationship: None,
            },
            RelationshipLevel::Neutral => DialogueResponse {
                text: "Hello. What brings you here today?".to_string(),
                tone: DialogueTone::Neutral,
                mercy_impact: 0.02,
                requires_relationship: None,
            },
            _ => DialogueResponse {
                text: "...".to_string(),
                tone: DialogueTone::Wary,
                mercy_impact: -0.05,
                requires_relationship: None,
            },
        }
    }
}