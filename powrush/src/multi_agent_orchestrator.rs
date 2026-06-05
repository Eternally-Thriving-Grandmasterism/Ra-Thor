//! POWRUSH-MMO Multi-Agent Orchestrator
//! v16.3-improved-emotional-contagion
//!
//! Production upgrade to Emotional Contagion:
//! - Weighted influence based on interaction history
//! - Arousal-gated spread (emotions spread more when source is highly aroused)
//! Fully integrated with Valence-Arousal model, goals, memory, and RBE.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

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
pub struct EmotionalState {
    pub valence: f32,
    pub arousal: f32,
}

impl EmotionalState {
    pub fn new() -> Self { Self { valence: 0.0, arousal: 0.5 } }
    pub fn decay(&mut self, amount: f32) { /* ... */ }
    pub fn apply_event(&mut self, valence_delta: f32, arousal_delta: f32) { /* ... */ }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityState {
    pub valence: f32,
    pub contribution_score: f32,
    pub harmony: f32,
    pub last_quest_tick: u64,
    pub completed_skills: Vec<EducationSkill>,
    pub recent_interactions: Vec<u64>,
    pub last_goal_progress: f32,
    pub emotional_state: EmotionalState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NpcGoal { /* existing ... */ }

// ==================== Main Orchestrator ====================

pub struct MultiAgentOrchestrator {
    entities: HashMap<u64, EntityType>,
    entity_states: HashMap<u64, EntityState>,
    active_quests: HashMap<u64, Quest>,
    npc_goals: HashMap<u64, NpcGoal>,
    next_quest_id: u64,
    next_id: u64,
    current_tick: u64,
}

impl MultiAgentOrchestrator {
    pub fn new() -> Self { /* ... */ }

    pub fn tick(&mut self, delta_seconds: f32) {
        self.current_tick += 1;

        for state in self.entity_states.values_mut() {
            state.harmony = (state.harmony * 0.995 + 0.005).clamp(0.5, 1.2);
            state.emotional_state.decay(0.02);
        }

        self.apply_emotional_contagion(); // Improved in v16.3
        self.run_autonomous_npc_behavior();
        self.apply_npc_rbe_impact();

        if self.current_tick % 40 == 0 {
            self.generate_world_event_quest();
        }
    }

    // ==================== v16.3: Improved Emotional Contagion ====================

    fn apply_emotional_contagion(&mut self) {
        let npc_ids: Vec<u64> = self.entities
            .iter()
            .filter(|(_, e)| matches!(e, EntityType::AiAgent { .. } | EntityType::AgiEntity { .. }))
            .map(|(id, _)| *id)
            .collect();

        for &id in &npc_ids {
            if let Some(state) = self.entity_states.get(&id) {
                let mut total_valence = state.emotional_state.valence;
                let mut total_arousal = state.emotional_state.arousal;
                let mut weight_sum = 1.0;

                for &other_id in &state.recent_interactions {
                    if let Some(other) = self.entity_states.get(&other_id) {
                        // Arousal gate: only spread if the other is emotionally "loud"
                        if other.emotional_state.arousal < 0.5 {
                            continue;
                        }

                        // Weighted influence: more interactions = stronger tie
                        let interaction_weight = (state.recent_interactions.len() as f32).min(5.0) / 5.0;
                        let influence = 0.12 * interaction_weight;

                        total_valence += other.emotional_state.valence * influence;
                        total_arousal += other.emotional_state.arousal * influence;
                        weight_sum += influence;
                    }
                }

                if weight_sum > 1.0 {
                    if let Some(my_state) = self.entity_states.get_mut(&id) {
                        my_state.emotional_state.valence = (total_valence / weight_sum).clamp(-1.0, 1.0);
                        my_state.emotional_state.arousal = (total_arousal / weight_sum).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡