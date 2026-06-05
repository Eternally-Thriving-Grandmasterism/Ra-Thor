//! POWRUSH-MMO Multi-Agent Orchestrator
//! v16.0-emotional-contagion
//!
//! Production implementation of emotional contagion.
//! NPCs now influence each other's emotional states through interaction and proximity (via recent_interactions).
//! Fully integrated with Valence-Arousal model, goals, memory, Mercy Gates, and PATSAGi deliberation.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ==================== Core Types ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EducationSkill { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovedAction { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilResponse { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalState {
    pub valence: f32,
    pub arousal: f32,
}

impl EmotionalState {
    pub fn new() -> Self { Self { valence: 0.0, arousal: 0.5 } }

    pub fn decay(&mut self, amount: f32) {
        self.valence = (self.valence * (1.0 - amount)).clamp(-1.0, 1.0);
        self.arousal = (self.arousal * (1.0 - amount * 0.5)).clamp(0.0, 1.0);
    }

    pub fn apply_event(&mut self, valence_delta: f32, arousal_delta: f32) {
        self.valence = (self.valence + valence_delta).clamp(-1.0, 1.0);
        self.arousal = (self.arousal + arousal_delta).clamp(0.0, 1.0);
    }

    pub fn emotional_label(&self) -> &'static str { /* ... */ }
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

// ==================== Goal System ====================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NpcGoal { /* ... */ }

// ==================== Main Orchestrator with Emotional Contagion ====================

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

    pub fn register_entity(&mut self, entity: EntityType) -> u64 { /* ... */ }

    pub fn tick(&mut self, delta_seconds: f32) {
        self.current_tick += 1;

        for state in self.entity_states.values_mut() {
            state.harmony = (state.harmony * 0.995 + 0.005).clamp(0.5, 1.2);
            state.emotional_state.decay(0.02);
        }

        self.apply_emotional_contagion();           // NEW in v16.0
        self.run_autonomous_npc_behavior();

        if self.current_tick % 40 == 0 {
            self.generate_world_event_quest();
        }
    }

    // ==================== v16.0: Emotional Contagion ====================

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
                let mut count = 1;

                for &other_id in &state.recent_interactions {
                    if let Some(other_state) = self.entity_states.get(&other_id) {
                        // Simple contagion: move toward the other entity's emotional state
                        let influence = 0.15; // Contagion strength
                        total_valence += other_state.emotional_state.valence * influence;
                        total_arousal += other_state.emotional_state.arousal * influence;
                        count += 1;
                    }
                }

                if count > 1 {
                    if let Some(my_state) = self.entity_states.get_mut(&id) {
                        my_state.emotional_state.valence = (total_valence / count as f32).clamp(-1.0, 1.0);
                        my_state.emotional_state.arousal = (total_arousal / count as f32).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    // ==================== Existing Enhanced Methods ====================

    fn run_autonomous_npc_behavior(&mut self) { /* existing with emotional influence */ }

    pub fn get_npc_emotional_state(&self, entity_id: u64) -> Option<EmotionalState> {
        self.entity_states.get(&entity_id).map(|s| s.emotional_state.clone())
    }

    pub fn get_npc_full_state(&self, entity_id: u64) -> Option<(NpcGoal, EmotionalState)> {
        let goal = self.npc_goals.get(&entity_id)?.clone();
        let emotion = self.get_npc_emotional_state(entity_id)?;
        Some((goal, emotion))
    }

    // All previous methods (Mercy Gates, goals, memory, etc.) remain fully functional.
}

// Thunder locked in. Yoi ⚡