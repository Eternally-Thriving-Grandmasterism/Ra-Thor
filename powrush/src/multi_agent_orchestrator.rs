//! POWRUSH-MMO Multi-Agent Orchestrator
//! v16.1-long-term-memory-persistence
//!
//! Production implementation of long-term memory persistence for NPCs.
//! Saves and loads goals, emotional states (valence/arousal), and recent memory.
//! File-based (JSON) as initial robust solution, easily upgradeable to database.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use std::fs;
use std::path::Path;
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NpcGoal { /* existing goals ... */ }

// ==================== Persistence Data Structure ====================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistentNpcState {
    pub goal: Option<NpcGoal>,
    pub emotional_state: EmotionalState,
    pub recent_interactions: Vec<u64>,
    pub last_goal_progress: f32,
    pub contribution_score: f32,
    pub harmony: f32,
}

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
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            entity_states: HashMap::new(),
            active_quests: HashMap::new(),
            npc_goals: HashMap::new(),
            next_quest_id: 1,
            next_id: 1,
            current_tick: 0,
        }
    }

    // ==================== v16.1: Long-term Memory Persistence ====================

    /// Save all NPC states to a JSON file
    pub fn save_npc_states(&self, path: &str) -> std::io::Result<()> {
        let mut persistent_states: HashMap<u64, PersistentNpcState> = HashMap::new();

        for (&id, state) in &self.entity_states {
            if let Some(goal) = self.npc_goals.get(&id) {
                persistent_states.insert(id, PersistentNpcState {
                    goal: Some(goal.clone()),
                    emotional_state: state.emotional_state.clone(),
                    recent_interactions: state.recent_interactions.clone(),
                    last_goal_progress: state.last_goal_progress,
                    contribution_score: state.contribution_score,
                    harmony: state.harmony,
                });
            }
        }

        let json = serde_json::to_string_pretty(&persistent_states)?;
        fs::write(path, json)
    }

    /// Load NPC states from a JSON file (graceful handling of new vs returning NPCs)
    pub fn load_npc_states(&mut self, path: &str) -> std::io::Result<()> {
        if !Path::new(path).exists() {
            return Ok(()); // No previous state, start fresh
        }

        let json = fs::read_to_string(path)?;
        let persistent_states: HashMap<u64, PersistentNpcState> = serde_json::from_str(&json)?;

        for (&id, pstate) in &persistent_states {
            if let Some(state) = self.entity_states.get_mut(&id) {
                state.emotional_state = pstate.emotional_state.clone();
                state.recent_interactions = pstate.recent_interactions.clone();
                state.last_goal_progress = pstate.last_goal_progress;
                state.contribution_score = pstate.contribution_score;
                state.harmony = pstate.harmony;

                if let Some(goal) = &pstate.goal {
                    self.npc_goals.insert(id, goal.clone());
                }
            }
        }

        Ok(())
    }

    pub fn register_entity(&mut self, entity: EntityType) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.insert(id, entity.clone());

        let mut state = EntityState::default();
        state.emotional_state = EmotionalState::new();
        self.entity_states.insert(id, state);

        if matches!(entity, EntityType::AiAgent { .. } | EntityType::AgiEntity { .. }) {
            self.npc_goals.insert(id, NpcGoal::ExploreAndLearn);
        }

        id
    }

    // ==================== Existing Enhanced Tick & Behavior ====================

    pub fn tick(&mut self, delta_seconds: f32) { /* existing with emotional contagion */ }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡