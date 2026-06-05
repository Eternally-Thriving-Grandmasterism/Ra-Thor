//! POWRUSH-MMO Multi-Agent Orchestrator
//! v17.1-experience-replay
//!
//! Production addition of Experience Replay Buffer for more stable Q-learning.
//! NPCs now sample past experiences for Q-updates instead of only learning from the latest transition.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::seq::SliceRandom;

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

// ==================== Q-Learning & Replay ====================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QValues { /* existing from v17.0 ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Experience {
    pub action_type: String,
    pub reward: f32,
    pub next_max_q: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NeuroSymbolicMemory {
    pub q_values: QValues,
    pub replay_buffer: Vec<Experience>,
    pub action_history: Vec<ActionOutcome>,
    pub learned_preference: f32,
}

const REPLAY_BUFFER_SIZE: usize = 128;
const REPLAY_BATCH_SIZE: usize = 8;

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

    // ==================== v17.1: Experience Replay ====================

    fn record_experience(&mut self, entity_id: u64, action: &Action, shaped_reward: f32, next_max_q: f32) {
        if let Some(memory) = self.neuro_memories.get_mut(&entity_id) {
            let exp = Experience {
                action_type: format!("{:?}", action),
                reward: shaped_reward,
                next_max_q,
            };

            if memory.replay_buffer.len() >= REPLAY_BUFFER_SIZE {
                memory.replay_buffer.remove(0); // FIFO
            }
            memory.replay_buffer.push(exp);

            // Occasionally perform replay updates
            if memory.replay_buffer.len() >= REPLAY_BATCH_SIZE && self.current_tick % 3 == 0 {
                self.perform_replay_update(entity_id);
            }
        }
    }

    fn perform_replay_update(&mut self, entity_id: u64) {
        if let Some(memory) = self.neuro_memories.get_mut(&entity_id) {
            if memory.replay_buffer.is_empty() { return; }

            let mut rng = rand::thread_rng();
            let batch: Vec<&Experience> = memory.replay_buffer
                .choose_multiple(&mut rng, REPLAY_BATCH_SIZE.min(memory.replay_buffer.len()))
                .collect();

            let alpha = 0.12;
            let gamma = 0.93;

            for exp in batch {
                let current_q = match exp.action_type.as_str() {
                    s if s.contains("Diplomacy") => &mut memory.q_values.diplomacy,
                    s if s.contains("Teach") => &mut memory.q_values.teach,
                    s if s.contains("Harvest") => &mut memory.q_values.harvest,
                    s if s.contains("ConsultCouncil") => &mut memory.q_values.consult_council,
                    s if s.contains("Create") => &mut memory.q_values.create,
                    _ => continue,
                };

                let target = exp.reward + gamma * exp.next_max_q;
                *current_q = *current_q + alpha * (target - *current_q);
                *current_q = current_q.clamp(-2.0, 4.0);
            }
        }
    }

    // Modified record_action_outcome to also record experience
    fn record_action_outcome(&mut self, entity_id: u64, action: &Action, shaped_reward: f32, eval: &MoralEvaluation) {
        let next_max_q = if let Some(memory) = self.neuro_memories.get(&entity_id) {
            memory.q_values.diplomacy
                .max(memory.q_values.teach)
                .max(memory.q_values.harvest)
                .max(memory.q_values.consult_council)
                .max(memory.q_values.create)
        } else { 0.0 };

        self.record_experience(entity_id, action, shaped_reward, next_max_q);
    }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡