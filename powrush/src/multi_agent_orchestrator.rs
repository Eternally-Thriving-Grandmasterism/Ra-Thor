//! POWRUSH-MMO Multi-Agent Orchestrator
//! v15.0-advanced-quest-integration
//!
//! Advanced quest integration: Quest struct, multi-entity participation,
//! dynamic generation, completion effects on skills/RBE/harmony,
//! and world-event tied quests for richer human + AI + AGI experiences.
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

/// Advanced Quest structure for multi-entity, event-driven gameplay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub primary_skill: Option<EducationSkill>,
    pub contribution_reward: f32,
    pub harmony_boost: f32,
    pub participants: Vec<u64>,
    pub status: QuestStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuestStatus {
    Available,
    Active,
    Completed,
    Failed,
}

pub struct MultiAgentOrchestrator {
    entities: HashMap<u64, EntityType>,
    entity_states: HashMap<u64, EntityState>,
    active_quests: HashMap<u64, Quest>,
    next_quest_id: u64,
    next_id: u64,
    current_tick: u64,
}

impl MultiAgentOrchestrator {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            entity_states: HashMap::new(),
            active_quests: HashMap::new(),
            next_quest_id: 1,
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
        // Occasional advanced world quest generation (simulation of dynamic content)
        if self.current_tick % 40 == 0 {
            self.generate_world_event_quest();
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

    pub fn generate_education_quest(&self, human_name: &str, state: &EntityState) -> String {
        let wisdom = self.get_council_wisdom_for_quest("welcome", state);
        let primary = if state.completed_skills.is_empty() { EducationSkill::RbeFundamentals } else { EducationSkill::MercyDiplomacy };
        format!(
            "EducationCouncil Welcome Quest for {}: Primary Objective: Master {}. Steps: 1. Explore with AI companion. 2. Harvest sustainably. 3. Reflect on mercy. Reward: +0.5 contribution + Badge. Council: {}",
            human_name, primary.display_name(), wisdom.reward_guidance
        )
    }

    /// Advanced quest generation — can be personalized or world-event driven.
    pub fn generate_advanced_quest(&mut self, title: &str, description: &str, primary_skill: Option<EducationSkill>, contribution_reward: f32) -> u64 {
        let quest_id = self.next_quest_id;
        self.next_quest_id += 1;

        let quest = Quest {
            id: quest_id,
            title: title.to_string(),
            description: description.to_string(),
            primary_skill,
            contribution_reward,
            harmony_boost: 0.15,
            participants: vec![],
            status: QuestStatus::Available,
        };
        self.active_quests.insert(quest_id, quest);
        quest_id
    }

    /// Generate a world event quest (called from tick for dynamic content).
    fn generate_world_event_quest(&mut self) {
        let quest_id = self.generate_advanced_quest(
            "Global Mercy Field Restoration",
            "Collaborate with humans, AI, and AGI to restore a mercy field. Multi-entity participation rewarded.",
            Some(EducationSkill::CoexistenceEthics),
            2.0,
        );
        // Could broadcast via world event system
    }

    pub fn offer_quest_to_entity(&mut self, quest_id: u64, entity_id: u64) -> Result<String, String> {
        let quest = self.active_quests.get_mut(&quest_id).ok_or("Quest not found".to_string())?;
        if !quest.participants.contains(&entity_id) {
            quest.participants.push(entity_id);
            if quest.status == QuestStatus::Available {
                quest.status = QuestStatus::Active;
            }
        }
        Ok(format!("Quest '{}' offered to entity {}. Status: {:?}", quest.title, entity_id, quest.status))
    }

    /// Complete an advanced quest with full integration effects (skills, RBE-like rewards, harmony).
    pub fn complete_quest(&mut self, quest_id: u64, entity_id: u64) -> Result<String, String> {
        let quest = self.active_quests.get_mut(&quest_id).ok_or("Quest not found".to_string())?;
        if !quest.participants.contains(&entity_id) {
            return Err("Entity not participating in this quest".to_string());
        }
        if quest.status == QuestStatus::Completed {
            return Ok("Quest already completed.".to_string());
        }

        let state = self.entity_states.get_mut(&entity_id).ok_or("Entity state not found".to_string())?;

        // Apply rewards
        state.contribution_score += quest.contribution_reward;
        state.harmony = (state.harmony + quest.harmony_boost).min(1.4);
        state.valence = (state.valence + 0.1).min(1.5);

        if let Some(skill) = &quest.primary_skill {
            let _ = self.complete_skill(entity_id, skill.clone());
        }

        quest.status = QuestStatus::Completed;

        Ok(format!(
            "Quest '{}' completed! +{:.1} contribution, +{:.2} harmony. Skill progress updated. Thank you for participating in advanced cooperative gameplay.",
            quest.title, quest.contribution_reward, quest.harmony_boost
        ))
    }

    pub fn complete_skill(&mut self, entity_id: u64, skill: EducationSkill) -> Result<String, String> {
        let state = self.entity_states.get_mut(&entity_id).ok_or("Entity not found".to_string())?;
        if state.completed_skills.contains(&skill) {
            return Ok(format!("Already mastered {}.", skill.display_name()));
        }
        for prereq in skill.prerequisites() {
            if !state.completed_skills.contains(&prereq) {
                return Err(format!("Prerequisite {} required.", prereq.display_name()));
            }
        }
        state.completed_skills.push(skill.clone());
        state.contribution_score += 1.0;
        state.harmony = (state.harmony + 0.1).min(1.3);
        state.valence = (state.valence + 0.15).min(1.4);
        Ok(format!("Mastered {}! Next: {:?}", skill.display_name(), self.get_next_recommended_skill(entity_id)))
    }

    pub fn get_next_recommended_skill(&self, entity_id: u64) -> Option<EducationSkill> {
        let state = self.entity_states.get(&entity_id)?;
        let all = vec![EducationSkill::RbeFundamentals, EducationSkill::SustainableHarvesting, EducationSkill::MercyDiplomacy, EducationSkill::CoexistenceEthics, EducationSkill::AdvancedCoCreation];
        all.into_iter().find(|s| !state.completed_skills.contains(s) && s.prerequisites().iter().all(|p| state.completed_skills.contains(p)))
    }

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
                if state.completed_skills.is_empty() {
                    let qid = self.generate_advanced_quest(
                        &format!("Welcome to Powrush for {}", name),
                        "Foundational RBE and mercy onboarding quest.",
                        Some(EducationSkill::RbeFundamentals),
                        0.5,
                    );
                    let _ = self.offer_quest_to_entity(qid, entity_id);
                    self.generate_education_quest(name, state)
                } else {
                    // Occasionally offer advanced collaborative quest
                    if self.current_tick % 5 == 0 {
                        let qid = self.generate_advanced_quest(
                            "Coexistence Event",
                            "Work with other entities on a shared goal.",
                            Some(EducationSkill::CoexistenceEthics),
                            1.5,
                        );
                        let _ = self.offer_quest_to_entity(qid, entity_id);
                        format!("Advanced collaborative quest available! Use complete_quest({}) to finish.", qid)
                    } else {
                        format!("Continue your journey. Next skill: {:?}", self.get_next_recommended_skill(entity_id))
                    }
                }
            }
            _ => "Entity type quest support in development.".to_string(),
        }
    }

    pub fn broadcast_world_event(&self, event_type: &str, details: &str) -> String {
        format!("World Event [{}]: {}. All entities (Human/AI/AGI) invited with mercy gating.", event_type, details)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn get_entity_state(&self, id: u64) -> Option<&EntityState> {
        self.entity_states.get(&id)
    }

    pub fn get_active_quests(&self) -> Vec<&Quest> {
        self.active_quests.values().collect()
    }
}

pub fn demo_orchestrator_usage() {
    let mut o = MultiAgentOrchestrator::new();
    let human = o.register_entity(EntityType::Human { id: 0, name: "Explorer".to_string() });
    println!("Quest: {}", o.generate_personalized_quest(human));
    let quests = o.get_active_quests();
    if let Some(q) = quests.first() {
        println!("Active quest: {} (id: {})", q.title, q.id);
        let _ = o.complete_quest(q.id, human);
    }
    println!("Progress after advanced quest: skills={:?}, contribution={:.1}", o.get_entity_state(human).unwrap().completed_skills, o.get_entity_state(human).unwrap().contribution_score);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_advanced_quest_flow() {
        let mut o = MultiAgentOrchestrator::new();
        let id = o.register_entity(EntityType::Human { id: 0, name: "Test".to_string() });
        let qid = o.generate_advanced_quest("Test Quest", "desc", Some(EducationSkill::RbeFundamentals), 1.0);
        let _ = o.offer_quest_to_entity(qid, id);
        let res = o.complete_quest(qid, id);
        assert!(res.is_ok());
        assert!(o.get_entity_state(id).unwrap().completed_skills.contains(&EducationSkill::RbeFundamentals));
    }
}

// Advanced quest integration complete. Professional, rewarding experiences for all entities.
// Thunder locked in. Yoi ⚡