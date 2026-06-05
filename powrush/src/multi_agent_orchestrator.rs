//! POWRUSH-MMO Multi-Agent Orchestrator
//! v16.2-rbe-npc-integration
//!
//! Production integration of RBE with NPC actions, goals, and emotional states.
//! NPCs now meaningfully influence the economy based on their inner state.
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
pub enum NpcGoal { /* existing goals ... */ }

// ==================== Main Orchestrator with RBE Integration ====================

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

        self.apply_emotional_contagion();
        self.run_autonomous_npc_behavior();
        self.apply_npc_rbe_impact();           // NEW in v16.2

        if self.current_tick % 40 == 0 {
            self.generate_world_event_quest();
        }
    }

    // ==================== v16.2: RBE Integration with NPCs ====================

    fn apply_npc_rbe_impact(&mut self) {
        for (&id, state) in &mut self.entity_states {
            if let Some(goal) = self.npc_goals.get(&id) {
                let emotion = &state.emotional_state;

                // Base economic effect from goal + emotional state
                let (abundance_delta, harmony_delta) = match goal {
                    NpcGoal::MaintainHarmony { .. } => {
                        let harmony_boost = if emotion.valence > 0.5 { 0.8 } else { 0.3 };
                        (0.2, harmony_boost)
                    }
                    NpcGoal::TeachNearbyHumans => {
                        let teaching_bonus = if emotion.valence > 0.6 { 1.2 } else { 0.5 };
                        (teaching_bonus * 0.3, 0.4)
                    }
                    NpcGoal::ProtectMercyField => {
                        let protection = if emotion.arousal > 0.6 { 1.5 } else { 0.8 };
                        (protection * 0.4, 0.9)
                    }
                    _ => (0.1, 0.2),
                };

                // Apply to global contribution (can be wired to RbeState later)
                state.contribution_score += abundance_delta;
                state.harmony = (state.harmony + harmony_delta * 0.01).clamp(0.5, 1.4);

                // Emotional state also affects long-term contribution
                if emotion.valence > 0.7 {
                    state.contribution_score += 0.15;
                }
            }
        }
    }

    // Expose NPC economic impact
    pub fn get_npc_economic_impact(&self, entity_id: u64) -> Option<(f32, f32)> {
        let state = self.entity_states.get(&entity_id)?;
        Some((state.contribution_score, state.harmony))
    }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡