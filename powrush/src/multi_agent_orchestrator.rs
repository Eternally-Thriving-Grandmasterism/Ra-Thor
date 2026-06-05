//! POWRUSH-MMO Multi-Agent Orchestrator
//! v17.0-activate-qlearning
//!
//! Production activation and refinement of Q-learning inside the autonomous NPC behavior loop.
//! NPCs now use learned Q-values to propose actions, while remaining under full Mercy + PATSAGi constraints.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::Rng; // For exploration

// ==================== Core Types ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EducationSkill { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovedAction { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilResponse { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalState { /* existing ... */ }

impl EmotionalState { /* existing methods ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityState { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NpcGoal { /* existing ... */ }

// ==================== Q-Learning & Memory ====================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QValues {
    pub diplomacy: f32,
    pub teach: f32,
    pub harvest: f32,
    pub consult_council: f32,
    pub create: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NeuroSymbolicMemory {
    pub q_values: QValues,
    pub action_history: Vec<ActionOutcome>,
    pub learned_preference: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionOutcome { /* existing ... */ }

// ==================== Main Orchestrator ====================

pub struct MultiAgentOrchestrator {
    entities: HashMap<u64, EntityType>,
    entity_states: HashMap<u64, EntityState>,
    active_quests: HashMap<u64, Quest>,
    npc_goals: HashMap<u64, NpcGoal>,
    neuro_memories: HashMap<u64, NeuroSymbolicMemory>,
    next_quest_id: u64,
    next_id: u64,
    current_tick: u64,
}

impl MultiAgentOrchestrator {
    pub fn new() -> Self { /* ... */ }

    pub fn register_entity(&mut self, entity: EntityType) -> u64 { /* ... */ }

    pub fn tick(&mut self, delta_seconds: f32) { /* existing */ }

    // ==================== v17.0: Q-Learning Activated in Behavior Loop ====================

    fn run_autonomous_npc_behavior(&mut self) {
        let npc_ids: Vec<u64> = self.entities
            .iter()
            .filter(|(_, e)| matches!(e, EntityType::AiAgent { .. } | EntityType::AgiEntity { .. }))
            .map(|(id, _)| *id)
            .collect();

        for &entity_id in &npc_ids {
            let current_goal = self.npc_goals.get(&entity_id).cloned();
            let emotional = self.entity_states.get(&entity_id)
                .map(|s| s.emotional_state.clone())
                .unwrap_or_default();

            // === Q-Learning influenced action proposal ===
            let proposed = if let Some(memory) = self.neuro_memories.get(&entity_id) {
                self.select_action_with_q_values(entity_id, current_goal.as_ref(), &emotional, memory)
            } else {
                // Fallback to goal-based selection
                self.fallback_goal_based_action(current_goal.as_ref(), &emotional)
            };

            let approved = self.decide_action_with_mercy_and_councils(entity_id, proposed);

            match approved {
                ApprovedAction::Execute(action) => {
                    self.execute_approved_npc_action(entity_id, action.clone());

                    // Record outcome and update Q-values
                    let moral_eval = self.evaluate_moral_reasoning(&action, entity_id);
                    let components = self.build_reward_components(entity_id, &action, &moral_eval);
                    let shaped_reward = compute_shaped_reward(&components);

                    self.update_q_values(entity_id, &action, shaped_reward);
                    self.record_action_outcome(entity_id, &action, shaped_reward, &moral_eval);
                }
                ApprovedAction::Transform { .. } => {
                    // Still learn from transformed actions
                    if let Some(memory) = self.neuro_memories.get_mut(&entity_id) {
                        memory.learned_preference = (memory.learned_preference - 0.03).max(0.0);
                    }
                }
                ApprovedAction::Block { .. } => {
                    // Negative learning signal
                    if let Some(memory) = self.neuro_memories.get_mut(&entity_id) {
                        memory.learned_preference = (memory.learned_preference - 0.05).max(0.0);
                    }
                }
            }
        }
    }

    fn select_action_with_q_values(
        &self,
        entity_id: u64,
        goal: Option<&NpcGoal>,
        emotional: &EmotionalState,
        memory: &NeuroSymbolicMemory,
    ) -> Action {
        let mut rng = rand::thread_rng();
        let epsilon = 0.12; // Exploration rate

        // ε-greedy exploration
        if rng.gen::<f32>() < epsilon {
            return self.fallback_goal_based_action(goal, emotional);
        }

        // Exploit best Q-value action
        let q = &memory.q_values;
        let best_action = if q.teach >= q.diplomacy && q.teach >= q.harvest && q.teach >= q.consult_council && q.teach >= q.create {
            Action::Teach { learner: 0, skill: "Learned from experience".to_string(), mercy_intent: 0.9 }
        } else if q.diplomacy >= q.harvest && q.diplomacy >= q.consult_council && q.diplomacy >= q.create {
            Action::Diplomacy { faction: "local".to_string(), proposal: "Learned diplomacy".to_string() }
        } else if q.harvest >= q.consult_council && q.harvest >= q.create {
            Action::Harvest { node: "mercy_field".to_string() }
        } else if q.consult_council >= q.create {
            Action::ConsultCouncil { council: "GeneralHarmonyCouncil".to_string(), query: "What is wise now?".to_string() }
        } else {
            Action::Create { blueprint: "experience_based".to_string(), resources: vec![] }
        };

        best_action
    }

    fn fallback_goal_based_action(&self, goal: Option<&NpcGoal>, emotional: &EmotionalState) -> Action {
        // Previous goal + emotion based logic
        match goal {
            Some(NpcGoal::MaintainHarmony { .. }) => Action::Diplomacy { faction: "local".to_string(), proposal: "Maintain balance".to_string() },
            Some(NpcGoal::TeachNearbyHumans) => Action::Teach { learner: 0, skill: "RBE & Mercy".to_string(), mercy_intent: 0.85 },
            _ => Action::Harvest { node: "mercy_field".to_string() },
        }
    }

    fn build_reward_components(&self, entity_id: u64, action: &Action, eval: &MoralEvaluation) -> RewardComponents { /* ... */ }

    fn update_q_values(&mut self, entity_id: u64, action: &Action, shaped_reward: f32) { /* existing Q-update from v16.9 */ }

    fn record_action_outcome(&mut self, entity_id: u64, action: &Action, shaped_reward: f32, eval: &MoralEvaluation) { /* ... */ }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡