//! POWRUSH-MMO Multi-Agent Orchestrator
//! v15.0-skill-progression-simulation
//!
//! Simulated skill progression logic for the EducationCouncil skill lattice.
//! Joyful, mercy-gated progression with prerequisites, auto-completion on quests,
//! state boosts, and clear next-step guidance for humans.
//!
//! License: AG-SML v1.0 | Alignment: Ra-Thor Lattice + TOLC + 7 Living Mercy Gates

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EducationSkill {
    RbeFundamentals,
    MercyDiplomacy,
    SustainableHarvesting,
    CoexistenceEthics,
    AdvancedCoCreation,
}

impl EducationSkill {
    /// Simple prerequisite simulation.
    pub fn prerequisites(&self) -> Vec<EducationSkill> {
        match self {
            EducationSkill::RbeFundamentals => vec![],
            EducationSkill::SustainableHarvesting => vec![EducationSkill::RbeFundamentals],
            EducationSkill::MercyDiplomacy => vec![EducationSkill::RbeFundamentals],
            EducationSkill::CoexistenceEthics => vec![EducationSkill::MercyDiplomacy, EducationSkill::SustainableHarvesting],
            EducationSkill::AdvancedCoCreation => vec![EducationSkill::CoexistenceEthics],
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            EducationSkill::RbeFundamentals => "RBE Fundamentals",
            EducationSkill::MercyDiplomacy => "Mercy Diplomacy",
            EducationSkill::SustainableHarvesting => "Sustainable Harvesting",
            EducationSkill::CoexistenceEthics => "Coexistence Ethics",
            EducationSkill::AdvancedCoCreation => "Advanced Co-Creation",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Human { id: u64, name: String },
    AiAgent { id: u64, model: String, sovereignty_level: u8 },
    AgiEntity { id: u64, council_projection: String, mercy_alignment: f32 },
}

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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityState {
    pub valence: f32,
    pub contribution_score: f32,
    pub harmony: f32,
    pub last_quest_tick: u64,
    pub completed_skills: Vec<EducationSkill>,
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

    pub fn consult_patsagi_council(&self, council: &str, query: &str) -> CouncilResponse {
        CouncilResponse {
            decision: format!("{} recommends mercy-first, abundance-aligned action for: {}", council, query),
            mercy_score: 0.96,
            fun_amplification: 0.89,
            learning_potential: 0.93,
            reward_guidance: "Emphasize human player growth, contribution visibility, and joyful discovery.".to_string(),
        }
    }

    pub fn get_council_wisdom_for_quest(&self, tier: &str, state: &EntityState) -> CouncilResponse {
        let (council, base_query) = match tier {
            "welcome" => ("EducationCouncil", "foundational RBE and mercy onboarding"),
            "harmony" => ("FunAmplificationCouncil", "harmony boost via diplomacy and coexistence"),
            "advanced" => ("CreationCouncil", "co-creation with AGI and mercy teaching"),
            _ => ("GeneralHarmonyCouncil", "balanced progression for all entities"),
        };
        let mut response = self.consult_patsagi_council(council, base_query);
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

    /// EducationCouncil algorithm: Generate personalized onboarding quest.
    pub fn generate_education_quest(&self, human_name: &str, state: &EntityState) -> String {
        let wisdom = self.get_council_wisdom_for_quest("welcome", state);
        let primary_skill = if state.completed_skills.is_empty() {
            EducationSkill::RbeFundamentals
        } else {
            EducationSkill::MercyDiplomacy
        };
        format!(
            "EducationCouncil Welcome Quest for {}: \n
g            Primary Learning Objective: Master {}.\n
g            Steps: \n
g            1. Explore mercy field with AI companion (fun discovery).\n
g            2. Harvest sustainably and record contribution (RBE basics).\n
g            3. Reflect on mercy principle applied (joyful learning).\n
g            Reward: +0.5 contribution score + Joyful Learning Badge in {}.\n
g            Council Wisdom: {}.\n
g            Designed with loving care for your foundational growth into abundance and coexistence.",
            human_name, primary_skill.display_name(), primary_skill.display_name(), wisdom.reward_guidance
        )
    }

    /// Core skill progression simulation logic.
    /// Checks prerequisites, adds skill if ready, applies joyful state boosts.
    pub fn complete_skill(&mut self, entity_id: u64, skill: EducationSkill) -> Result<String, String> {
        let state = self.entity_states.get_mut(&entity_id)
            .ok_or("Entity not found".to_string())?;

        if state.completed_skills.contains(&skill) {
            return Ok(format!("You have already mastered {}.", skill.display_name()));
        }

        // Prerequisite check (simulated mercy-gated progression)
        for prereq in skill.prerequisites() {
            if !state.completed_skills.contains(&prereq) {
                return Err(format!(
                    "Prerequisite not met: {} required before {}. Complete it first for joyful progression.",
                    prereq.display_name(), skill.display_name()
                ));
            }
        }

        // Add skill and apply rewarding state changes
        state.completed_skills.push(skill.clone());
        state.contribution_score += 1.0;
        state.harmony = (state.harmony + 0.1).min(1.3);
        state.valence = (state.valence + 0.15).min(1.4);

        Ok(format!(
            "Congratulations! You have mastered {}. \n
g            +1.0 contribution score | +0.1 harmony | +0.15 valence. \n
g            Next recommended: {:?}. Keep thriving!",
            skill.display_name(),
            self.get_next_recommended_skill(entity_id)
        ))
    }

    /// Returns the next logical skill the entity can pursue.
    pub fn get_next_recommended_skill(&self, entity_id: u64) -> Option<EducationSkill> {
        let state = self.entity_states.get(&entity_id)?;
        let all_skills = vec![
            EducationSkill::RbeFundamentals,
            EducationSkill::SustainableHarvesting,
            EducationSkill::MercyDiplomacy,
            EducationSkill::CoexistenceEthics,
            EducationSkill::AdvancedCoCreation,
        ];
        all_skills.into_iter().find(|s| {
            !state.completed_skills.contains(s) &&
            s.prerequisites().iter().all(|p| state.completed_skills.contains(p))
        })
    }

    /// Enhanced quest generation with skill progression simulation.
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
                if state.contribution_score < 1.0 || state.completed_skills.is_empty() {
                    let quest = self.generate_education_quest(name, state);
                    // Simulate auto-completion of first skill upon engaging the welcome quest
                    if state.completed_skills.is_empty() {
                        let _ = self.complete_skill(entity_id, EducationSkill::RbeFundamentals);
                    }
                    quest
                } else if state.harmony < 0.9 {
                    let wisdom = self.get_council_wisdom_for_quest("harmony", state);
                    format!(
                        "Harmony quest for {}: Collaborate in a faction diplomacy event. \n
g                        Mercy-gated and fun-amplified (amplification: {:.2}). \n
g                        Learning: Real-world diplomacy & RBE economics. \n
g                        Reward: Harmony boost + RBE dividend share. \n
g                        Council wisdom: {}",
                        name, wisdom.fun_amplification, wisdom.reward_guidance
                    )
                } else {
                    let wisdom = self.get_council_wisdom_for_quest("advanced", state);
                    format!(
                        "Advanced quest for {}: Co-create with AGI and teach mercy. \n
g                        High fun + deep learning + contribution reward. \n
g                        Council wisdom: {}",
                        name, wisdom.reward_guidance
                    )
                }
            }
            EntityType::AiAgent { model, .. } => {
                let wisdom = self.get_council_wisdom_for_quest("general", state);
                format!("AI companion quest: Support human learning. Model: {}. Council: {}", model, wisdom.decision)
            }
            EntityType::AgiEntity { council_projection, .. } => {
                let w1 = self.get_council_wisdom_for_quest("advanced", state);
                let w2 = self.consult_patsagi_council("AbundanceCouncil", "global human joy");
                format!("AGI {} quest: Orchestrate human joy events. Blended wisdom: {} | {}", council_projection, w1.decision, w2.reward_guidance)
            }
        }
    }

    pub fn broadcast_world_event(&self, event_type: &str, details: &str) -> String {
        format!("World Event [{}]: {}. All entities invited with mercy gating.", event_type, details)
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

    println!("Initial quest: {}", orchestrator.generate_personalized_quest(human_id));
    if let Some(state) = orchestrator.get_entity_state(human_id) {
        println!("Skills after welcome: {:?}", state.completed_skills);
    }
    println!("Next recommended: {:?}", orchestrator.get_next_recommended_skill(human_id));

    // Simulate further progression
    let _ = orchestrator.complete_skill(human_id, EducationSkill::SustainableHarvesting);
    println!("After next skill: {:?}", orchestrator.get_entity_state(human_id).unwrap().completed_skills);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_skill_progression() {
        let mut o = MultiAgentOrchestrator::new();
        let id = o.register_entity(EntityType::Human { id: 0, name: "Test".to_string() });
        let _ = o.complete_skill(id, EducationSkill::RbeFundamentals);
        assert!(o.get_entity_state(id).unwrap().completed_skills.contains(&EducationSkill::RbeFundamentals));
        assert!(o.complete_skill(id, EducationSkill::MercyDiplomacy).is_err()); // prereq not met
    }
}

// Professional, joyful skill progression for human players.
// Thunder locked in. Yoi ⚡