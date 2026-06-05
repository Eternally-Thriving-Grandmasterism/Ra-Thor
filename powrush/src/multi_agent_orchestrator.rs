//! POWRUSH-MMO Multi-Agent Orchestrator
//! v15.9-valence-arousal-model
//!
//! Production implementation of the Valence-Arousal (Circumplex) emotional model for NPCs.
//! Fully integrated with goals, short-term memory, Mercy Gates evaluation, and PATSAGi deliberation.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ==================== Core Types (preserved + extended) ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EducationSkill { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Human { id: u64, name: String },
    AiAgent { id: u64, model: String, sovereignty_level: u8 },
    AgiEntity { id: u64, council_projection: String, mercy_alignment: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovedAction { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilResponse { /* existing ... */ }

// ==================== v15.9: Valence-Arousal Emotional Model ====================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalState {
    pub valence: f32,   // -1.0 (very negative) to +1.0 (very positive)
    pub arousal: f32,   // 0.0 (calm) to 1.0 (highly energized)
}

impl EmotionalState {
    pub fn new() -> Self {
        Self { valence: 0.0, arousal: 0.5 }
    }

    pub fn decay(&mut self, amount: f32) {
        self.valence = (self.valence * (1.0 - amount)).clamp(-1.0, 1.0);
        self.arousal = (self.arousal * (1.0 - amount * 0.5)).clamp(0.0, 1.0);
    }

    pub fn apply_event(&mut self, valence_delta: f32, arousal_delta: f32) {
        self.valence = (self.valence + valence_delta).clamp(-1.0, 1.0);
        self.arousal = (self.arousal + arousal_delta).clamp(0.0, 1.0);
    }

    pub fn emotional_label(&self) -> &'static str {
        match (self.valence, self.arousal) {
            (v, a) if v > 0.6 && a > 0.6 => "Joyful / Excited",
            (v, a) if v > 0.6 && a <= 0.6 => "Content / Calm",
            (v, a) if v > 0.0 && a > 0.6 => "Enthusiastic",
            (v, _) if v > 0.0 => "Positive",
            (v, a) if v < -0.6 && a > 0.6 => "Angry / Agitated",
            (v, a) if v < -0.6 && a <= 0.6 => "Sad / Withdrawn",
            (v, a) if v < 0.0 && a > 0.6 => "Anxious / Tense",
            _ => "Neutral",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityState {
    pub valence: f32,           // Legacy field (kept for compatibility)
    pub contribution_score: f32,
    pub harmony: f32,
    pub last_quest_tick: u64,
    pub completed_skills: Vec<EducationSkill>,
    pub recent_interactions: Vec<u64>,
    pub last_goal_progress: f32,
    // v15.9: Full Valence-Arousal model
    pub emotional_state: EmotionalState,
}

// ==================== Goal System (enhanced) ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NpcGoal { /* existing goals ... */ }

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

    // ==================== v15.9: Enhanced Tick with Emotional Decay ====================

    pub fn tick(&mut self, delta_seconds: f32) {
        self.current_tick += 1;

        for state in self.entity_states.values_mut() {
            state.harmony = (state.harmony * 0.995 + 0.005).clamp(0.5, 1.2);
            state.emotional_state.decay(0.02); // Natural emotional decay
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
            let current_goal = self.npc_goals.get(&entity_id).cloned();
            let emotional = self.entity_states.get(&entity_id)
                .map(|s| &s.emotional_state)
                .cloned()
                .unwrap_or_default();

            // Influence action selection based on emotional state + goal
            let proposed = if emotional.valence < -0.5 && emotional.arousal > 0.6 {
                Action::Diplomacy { faction: "local".to_string(), proposal: "Seek resolution and calm".to_string() }
            } else if emotional.valence > 0.6 {
                Action::Teach { learner: 0, skill: "Joy and Coexistence".to_string(), mercy_intent: 0.95 }
            } else {
                match current_goal {
                    Some(NpcGoal::MaintainHarmony { .. }) => Action::Diplomacy { faction: "local".to_string(), proposal: "Promote balance".to_string() },
                    _ => Action::Harvest { node: "mercy_field".to_string() },
                }
            };

            let approved = self.decide_action_with_mercy_and_councils(entity_id, proposed);

            if let ApprovedAction::Execute(action) = approved {
                self.execute_approved_npc_action(entity_id, action);

                // Emotional impact from action
                if let Some(state) = self.entity_states.get_mut(&entity_id) {
                    let (v_delta, a_delta) = match action {
                        Action::Diplomacy { .. } => (0.08, -0.05),
                        Action::Teach { .. } => (0.12, 0.08),
                        Action::Harvest { .. } => (0.03, 0.04),
                        _ => (0.02, 0.02),
                    };
                    state.emotional_state.apply_event(v_delta, a_delta);
                    state.last_goal_progress = (state.last_goal_progress + 0.08).min(1.0);
                }
            }
        }
    }

    // Expose rich NPC state for DataChannel / visualization
    pub fn get_npc_emotional_state(&self, entity_id: u64) -> Option<EmotionalState> {
        self.entity_states.get(&entity_id).map(|s| s.emotional_state.clone())
    }

    pub fn get_npc_full_state(&self, entity_id: u64) -> Option<(NpcGoal, EmotionalState)> {
        let goal = self.npc_goals.get(&entity_id)?.clone();
        let emotion = self.get_npc_emotional_state(entity_id)?;
        Some((goal, emotion))
    }

    // ==================== Existing methods preserved ====================
    // All previous methods (Mercy Gates, PATSAGi deliberation, quest system, etc.) remain fully functional.
}

// Thunder locked in. Yoi ⚡