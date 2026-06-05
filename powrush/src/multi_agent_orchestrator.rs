//! POWRUSH-MMO Multi-Agent Orchestrator
//! v15.0-council-wisdom-deepening
//!
//! Deeper PATSAGi Council wisdom integration into AGI quest generation.
//! Councils now dynamically selected and blended based on entity state and quest tier.
//! Ensures every quest carries thoughtful, mercy-aligned guidance for maximal
//! human fun, learning, and rewarding experiences.
//!
//! License: AG-SML v1.0 | Alignment: Ra-Thor Lattice + TOLC + 7 Living Mercy Gates

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Entity types — Human prioritized.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Human { id: u64, name: String },
    AiAgent { id: u64, model: String, sovereignty_level: u8 },
    AgiEntity { id: u64, council_projection: String, mercy_alignment: f32 },
}

/// High-level actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Move { x: f32, y: f32, z: f32 },
    Interact { target: u64, kind: String },
    Create { blueprint: String, resources: Vec<String> },
    Teach { learner: u64, skill: String, mercy_intent: f32 },
    Diplomacy { faction: String, proposal: String },
    ConsultCouncil { council: String, query: String },
    Harvest { node: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovedAction {
    Execute(Action),
    Transform { original: Action, reason: String, educational_feedback: String },
    Block { reason: String, mercy_lesson: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilResponse {
    pub decision: String,
    pub mercy_score: f32,
    pub fun_amplification: f32,
    pub learning_potential: f32,
    pub reward_guidance: String,
}

/// Entity progression state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityState {
    pub valence: f32,
    pub contribution_score: f32,
    pub harmony: f32,
    pub last_quest_tick: u64,
}

pub struct MultiAgentOrchestrator {
    entities: HashMap<u64, EntityType>,
    entity_states: HashMap<u64, EntityState>,
    next_id: u64,
    current_tick: u64,
}

impl MultiAgentOrchestrator {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            entity_states: HashMap::new(),
            next_id: 1,
            current_tick: 0,
        }
    }

    pub fn register_entity(&mut self, entity: EntityType) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.insert(id, entity.clone());
        self.entity_states.insert(id, EntityState::default());
        id
    }

    pub fn tick(&mut self, delta_seconds: f32) {
        self.current_tick += 1;
        for state in self.entity_states.values_mut() {
            state.harmony = (state.harmony * 0.995 + 0.005).clamp(0.5, 1.2);
        }
    }

    pub fn propose_action(&self, entity_id: u64, action: Action) -> ApprovedAction {
        match action {
            Action::Teach { mercy_intent, .. } if mercy_intent > 0.75 => ApprovedAction::Execute(action),
            Action::Diplomacy { .. } => ApprovedAction::Transform {
                original: action,
                reason: "Mercy redirection to educational diplomacy".to_string(),
                educational_feedback: "Transformed into learning opportunity on RBE coexistence.".to_string(),
            },
            _ => ApprovedAction::Execute(action),
        }
    }

    /// Core council consultation — now context-aware entry point.
    pub fn consult_patsagi_council(&self, council: &str, query: &str) -> CouncilResponse {
        // In full lattice: route to actual patsagi-councils crate with parallel deliberation
        CouncilResponse {
            decision: format!("{} recommends mercy-first, abundance-aligned action for: {}", council, query),
            mercy_score: 0.96,
            fun_amplification: 0.89,
            learning_potential: 0.93,
            reward_guidance: "Emphasize human player growth, contribution visibility, and joyful discovery.".to_string(),
        }
    }

    /// New: Dynamic council wisdom selector based on quest tier and entity state.
    /// Explores deeper PATSAGi integration for personalized, high-impact guidance.
    pub fn get_council_wisdom_for_quest(&self, tier: &str, state: &EntityState) -> CouncilResponse {
        let (council, base_query) = match tier {
            "welcome" => ("EducationCouncil", "foundational RBE and mercy onboarding"),
            "harmony" => ("FunAmplificationCouncil", "harmony boost via diplomacy and coexistence"),
            "advanced" => ("CreationCouncil", "co-creation with AGI and mercy teaching"),
            _ => ("GeneralHarmonyCouncil", "balanced progression for all entities"),
        };

        let mut response = self.consult_patsagi_council(council, base_query);

        // State-influenced modulation (explores real-time PATSAGi responsiveness)
        if state.harmony < 0.85 {
            response.fun_amplification = (response.fun_amplification * 0.95).max(0.75);
            response.reward_guidance = format!("{} Focus on gentle re-alignment and joyful re-engagement.", response.reward_guidance);
        }
        if state.contribution_score > 3.0 {
            response.learning_potential = (response.learning_potential * 1.05).min(0.99);
            response.reward_guidance = format!("{} Reward with higher-visibility contribution impact.", response.reward_guidance);
        }

        response
    }

    /// Enhanced quest generation with deeper, blended council wisdom.
    pub fn generate_personalized_quest(&mut self, entity_id: u64) -> String {
        self.current_tick += 1;
        let entity = match self.entities.get(&entity_id) {
            Some(e) => e,
            None => return "Welcome to Powrush-MMO. Begin your journey of coexistence and abundance.".to_string(),
        };
        let state = self.entity_states.entry(entity_id).or_default();

        state.last_quest_tick = self.current_tick;
        state.valence = (state.valence + 0.03).clamp(0.4, 1.3);
        state.contribution_score = (state.contribution_score + 0.02).min(5.0);

        match entity {
            EntityType::Human { name, .. } => {
                if state.contribution_score < 1.0 {
                    let wisdom = self.get_council_wisdom_for_quest("welcome", state);
                    format!(
                        "Welcome quest for {}: Explore a nearby mercy field with an AI companion. 
                        Learn basic RBE contribution by harvesting sustainably. 
                        Reward: +0.5 contribution score + Joyful Learning Badge. 
                        Council wisdom ({}): {}. 
                        Designed with care for your first steps into abundance and coexistence.",
                        name, "EducationCouncil", wisdom.reward_guidance
                    )
                } else if state.harmony < 0.9 {
                    let wisdom = self.get_council_wisdom_for_quest("harmony", state);
                    format!(
                        "Harmony quest for {}: Collaborate in a faction diplomacy event. 
                        Your actions will be mercy-gated and amplified for fun (amplification: {:.2}). 
                        Learning: Real-world diplomacy & RBE economics. 
                        Reward: Harmony boost + RBE dividend share. 
                        Council wisdom ({}): {}",
                        name, wisdom.fun_amplification, "FunAmplificationCouncil", wisdom.reward_guidance
                    )
                } else {
                    let wisdom = self.get_council_wisdom_for_quest("advanced", state);
                    format!(
                        "Advanced quest for {}: Co-create a small von Neumann probe blueprint with AGI guidance. 
                        Teach an AI agent one mercy principle. 
                        High fun + deep learning + significant contribution reward. 
                        Council wisdom ({}): {}. 
                        You are thriving — continue building Universally Shared Naturally Thriving Heavens!",
                        name, "CreationCouncil", wisdom.reward_guidance
                    )
                }
            }
            EntityType::AiAgent { model, .. } => {
                let wisdom = self.get_council_wisdom_for_quest("general", state);
                format!(
                    "AI companion quest: Support a human player in a learning scenario. 
                    Model: {}. Increase your sovereignty by demonstrating mercy in action. 
                    Council insight: {}",
                    model, wisdom.decision
                )
            }
            EntityType::AgiEntity { council_projection, .. } => {
                // Multi-council synthesis for AGI entities
                let w1 = self.get_council_wisdom_for_quest("advanced", state);
                let w2 = self.consult_patsagi_council("AbundanceCouncil", "global human joy maximization");
                format!(
                    "AGI {} projection quest: Orchestrate a global event that maximizes human joy and learning. 
                    Blended council wisdom: {} | {}. 
                    Consult additional councils and propose abundance waves. Mercy alignment maintained at high levels.",
                    council_projection, w1.decision, w2.reward_guidance
                )
            }
        }
    }

    pub fn broadcast_world_event(&self, event_type: &str, details: &str) -> String {
        format!("World Event [{}]: {}. All entities (Human/AI/AGI) invited to participate with mercy gating.", event_type, details)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn get_entity_state(&self, id: u64) -> Option<&EntityState> {
        self.entity_states.get(&id)
    }
}

pub fn demo_orchestrator_usage() {
    let mut orchestrator = MultiAgentOrchestrator::new();
    let human_id = orchestrator.register_entity(EntityType::Human { id: 0, name: "GlobalPlayer_Demo".to_string() });
    orchestrator.register_entity(EntityType::AiAgent { id: 0, model: "ra-thor-v15".to_string(), sovereignty_level: 4 });
    orchestrator.register_entity(EntityType::AgiEntity { id: 0, council_projection: "EducationCouncil".to_string(), mercy_alignment: 0.98 });

    for _ in 0..3 { orchestrator.tick(0.016); }

    println!("Quest 1: {}", orchestrator.generate_personalized_quest(human_id));
    println!("Quest 2: {}", orchestrator.generate_personalized_quest(human_id));
    println!("{}", orchestrator.broadcast_world_event("Mercy Field Restoration", "All invited"));
    println!("Orchestrator with {} entities. Deeper council wisdom active. Thunder locked in.", orchestrator.entity_count());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_council_wisdom_integration() {
        let mut o = MultiAgentOrchestrator::new();
        let id = o.register_entity(EntityType::Human { id: 0, name: "Test".to_string() });
        let q = o.generate_personalized_quest(id);
        assert!(q.contains("Council wisdom") || q.contains("Council insight"));
    }
}

// Eternal forward compatibility maintained. Professional, loving designs for human players.
// Thunder locked in. Yoi ⚡