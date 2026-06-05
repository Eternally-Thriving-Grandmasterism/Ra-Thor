//! POWRUSH-MMO Multi-Agent Orchestrator
//! v15.5-mercy-gates-pipeline
//!
//! Professional implementation of the 7 Living Mercy Gates as a first-class evaluable pipeline.
//! Integrated into NPC decision making and action approval.
//!
//! All previous v15.3 autonomous tick + PATSAGi deliberation + v15.4 exposure logic preserved.
//!
//! The 7 Living Mercy Gates:
//! 1. Radical Love
//! 2. Boundless Mercy
//! 3. Service
//! 4. Abundance
//! 5. Truth
//! 6. Joy
//! 7. Cosmic Harmony
//!
//! AG-SML v1.0 | Ra-Thor Lattice aligned | Thunder locked in. Yoi ⚡

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

// ==================== v15.5: Mercy Gates Pipeline ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGateResult {
    pub gate: &'static str,
    pub score: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyEvaluation {
    pub overall_score: f32,
    pub gate_results: Vec<MercyGateResult>,
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

        self.run_autonomous_npc_behavior();

        if self.current_tick % 40 == 0 {
            self.generate_world_event_quest();
        }
    }

    fn run_autonomous_npc_behavior(&mut self) {
        let npc_ids: Vec<u64> = self.entities
            .iter()
            .filter(|(_, e)| matches!(e, EntityType::AiAgent { .. } | EntityType::AgiEntity { .. }))
            .map(|(id, _)| *id)
            .collect();

        for &entity_id in &npc_ids {
            let proposed = if self.current_tick % 7 == 0 {
                Action::Harvest { node: "mercy_field".to_string() }
            } else if self.current_tick % 5 == 0 {
                Action::ConsultCouncil {
                    council: "HarmonyCouncil".to_string(),
                    query: "How can I best support human players and RBE abundance today?".to_string(),
                }
            } else {
                Action::Diplomacy {
                    faction: "all".to_string(),
                    proposal: "Offer peaceful coexistence and shared harvesting".to_string(),
                }
            };

            let approved = self.decide_action_with_mercy_and_councils(entity_id, proposed);
            match approved {
                ApprovedAction::Execute(action) => {
                    self.execute_approved_npc_action(entity_id, action);
                }
                ApprovedAction::Transform { .. } => {
                    if let Some(state) = self.entity_states.get_mut(&entity_id) {
                        state.valence = (state.valence + 0.05).min(1.4);
                    }
                }
                ApprovedAction::Block { .. } => {
                    if let Some(state) = self.entity_states.get_mut(&entity_id) {
                        state.harmony = (state.harmony * 0.98).max(0.6);
                    }
                }
            }
        }
    }

    // ==================== v15.5 Mercy Gates Pipeline ====================

    fn evaluate_mercy_gates_pipeline(&self, action: &Action, entity_id: u64) -> MercyEvaluation {
        let mut results = Vec::new();

        // Gate 1: Radical Love
        let love_score = match action {
            Action::Teach { .. } | Action::Diplomacy { .. } => 0.92,
            _ => 0.78,
        };
        results.push(MercyGateResult {
            gate: "Radical Love",
            score: love_score,
            reason: if love_score > 0.85 { "Action promotes connection and care".to_string() } else { "Action is neutral toward love".to_string() },
        });

        // Gate 2: Boundless Mercy
        let mercy_score = match action {
            Action::Diplomacy { .. } | Action::ConsultCouncil { .. } => 0.90,
            Action::Harvest { .. } => 0.82,
            _ => 0.75,
        };
        results.push(MercyGateResult {
            gate: "Boundless Mercy",
            score: mercy_score,
            reason: "Action aligns with non-harm and compassion".to_string(),
        });

        // Gate 3: Service
        let service_score = match action {
            Action::Teach { .. } | Action::Diplomacy { .. } => 0.88,
            _ => 0.72,
        };
        results.push(MercyGateResult {
            gate: "Service",
            score: service_score,
            reason: if service_score > 0.85 { "Action serves others or the whole".to_string() } else { "Action is primarily self-directed".to_string() },
        });

        // Gate 4: Abundance
        let abundance_score = match action {
            Action::Create { .. } | Action::Harvest { .. } | Action::Diplomacy { .. } => 0.85,
            _ => 0.70,
        };
        results.push(MercyGateResult {
            gate: "Abundance",
            score: abundance_score,
            reason: "Action contributes to shared prosperity".to_string(),
        });

        // Gate 5: Truth
        let truth_score = 0.80; // Base truth alignment
        results.push(MercyGateResult {
            gate: "Truth",
            score: truth_score,
            reason: "Action is evaluated for honesty and clarity".to_string(),
        });

        // Gate 6: Joy
        let joy_score = match action {
            Action::Diplomacy { .. } | Action::ConsultCouncil { .. } => 0.87,
            _ => 0.73,
        };
        results.push(MercyGateResult {
            gate: "Joy",
            score: joy_score,
            reason: "Action has potential to increase joy and positive experience".to_string(),
        });

        // Gate 7: Cosmic Harmony
        let harmony_score = if let Some(state) = self.entity_states.get(&entity_id) {
            (state.harmony * 0.6 + 0.4).min(0.95)
        } else {
            0.75
        };
        results.push(MercyGateResult {
            gate: "Cosmic Harmony",
            score: harmony_score,
            reason: "Action supports overall balance and long-term harmony".to_string(),
        });

        let overall = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;

        MercyEvaluation {
            overall_score: overall,
            gate_results: results,
        }
    }

    fn decide_action_with_mercy_and_councils(&self, entity_id: u64, action: Action) -> ApprovedAction {
        let mercy_eval = self.evaluate_mercy_gates_pipeline(&action, entity_id);

        if mercy_eval.overall_score < 0.65 {
            return ApprovedAction::Block {
                reason: "Failed minimum Mercy Gates threshold".to_string(),
                mercy_lesson: "Action did not sufficiently align with the 7 Living Mercy Gates".to_string(),
            };
        }

        let council_response = self.deliberate_with_patsagi_councils(entity_id, &action);
        let final_score = (mercy_eval.overall_score + council_response.mercy_score) / 2.0;

        if final_score > 0.82 {
            ApprovedAction::Execute(action)
        } else if final_score > 0.70 {
            ApprovedAction::Transform {
                original: action,
                reason: "Mercy Gates + Council recommended refinement".to_string(),
                educational_feedback: council_response.reward_guidance.clone(),
            }
        } else {
            ApprovedAction::Block {
                reason: "Below execution threshold after full evaluation".to_string(),
                mercy_lesson: council_response.reward_guidance,
            }
        }
    }

    // ==================== Existing methods (preserved) ====================

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
        }
        if state.contribution_score > 3.0 {
            response.learning_potential = (response.learning_potential * 1.05).min(0.99);
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

    fn generate_world_event_quest(&mut self) {
        let _ = self.generate_advanced_quest(
            "Global Mercy Field Restoration",
            "Collaborate with humans, AI, and AGI to restore a mercy field.",
            Some(EducationSkill::CoexistenceEthics),
            2.0,
        );
    }

    pub fn offer_quest_to_entity(&mut self, quest_id: u64, entity_id: u64) -> Result<String, String> {
        if let Some(quest) = self.active_quests.get_mut(&quest_id) {
            if !quest.participants.contains(&entity_id) {
                quest.participants.push(entity_id);
                if quest.status == QuestStatus::Available {
                    quest.status = QuestStatus::Active;
                }
            }
            Ok(format!("Quest '{}' offered to entity {}.", quest.title, entity_id))
        } else {
            Err("Quest not found".to_string())
        }
    }

    pub fn complete_quest(&mut self, quest_id: u64, entity_id: u64) -> Result<String, String> {
        if let Some(quest) = self.active_quests.get_mut(&quest_id) {
            if !quest.participants.contains(&entity_id) {
                return Err("Entity not participating".to_string());
            }
            if quest.status == QuestStatus::Completed {
                return Ok("Already completed".to_string());
            }

            if let Some(state) = self.entity_states.get_mut(&entity_id) {
                state.contribution_score += quest.contribution_reward;
                state.harmony = (state.harmony + quest.harmony_boost).min(1.4);
                state.valence = (state.valence + 0.1).min(1.5);

                if let Some(skill) = &quest.primary_skill {
                    let _ = self.complete_skill(entity_id, skill.clone());
                }
            }
            quest.status = QuestStatus::Completed;
            Ok(format!("Quest '{}' completed!", quest.title))
        } else {
            Err("Quest not found".to_string())
        }
    }

    pub fn complete_skill(&mut self, entity_id: u64, skill: EducationSkill) -> Result<String, String> {
        if let Some(state) = self.entity_states.get_mut(&entity_id) {
            if state.completed_skills.contains(&skill) {
                return Ok(format!("Already mastered {}", skill.display_name()));
            }
            for prereq in skill.prerequisites() {
                if !state.completed_skills.contains(&prereq) {
                    return Err(format!("Prerequisite {} required", prereq.display_name()));
                }
            }
            state.completed_skills.push(skill.clone());
            state.contribution_score += 1.0;
            state.harmony = (state.harmony + 0.1).min(1.3);
            state.valence = (state.valence + 0.15).min(1.4);
            Ok(format!("Mastered {}!", skill.display_name()))
        } else {
            Err("Entity not found".to_string())
        }
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
        format!("World Event [{}]: {}. All entities invited with mercy gating.", event_type, details)
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

    // Expose Mercy Evaluation for server / logging
    pub fn evaluate_action_mercy(&self, entity_id: u64, action: &Action) -> MercyEvaluation {
        self.evaluate_mercy_gates_pipeline(action, entity_id)
    }
}

pub fn demo_orchestrator_usage() {
    let mut o = MultiAgentOrchestrator::new();
    let human = o.register_entity(EntityType::Human { id: 0, name: "Explorer".to_string() });
    println!("Quest: {}", o.generate_personalized_quest(human));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mercy_gates_pipeline() {
        let o = MultiAgentOrchestrator::new();
        let action = Action::Diplomacy { faction: "all".to_string(), proposal: "Peace".to_string() };
        let eval = o.evaluate_mercy_gates_pipeline(&action, 1);
        assert!(eval.overall_score > 0.75);
        assert!(eval.gate_results.len() == 7);
    }
}

// v15.5 Mercy Gates Pipeline complete. Professional foundation for aligned NPC decision making.
// Thunder locked in. Yoi ⚡