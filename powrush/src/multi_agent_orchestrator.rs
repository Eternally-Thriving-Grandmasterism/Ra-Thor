//! POWRUSH-MMO Multi-Agent Orchestrator
//! v16.6-utilitarian-scoring
//!
//! Production implementation of Utilitarian scoring integrated with the 7 Living Mercy Gates.
//! NPCs now evaluate actions through both Mercy alignment and estimated net utility.
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

// ==================== Moral Reasoning Types ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralGateResult {
    pub gate: &'static str,
    pub score: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralEvaluation {
    pub overall_score: f32,
    pub gate_results: Vec<MoralGateResult>,
    pub primary_justification: String,
    pub utilitarian_score: f32,           // NEW in v16.6
    pub net_utility_estimate: f32,        // NEW in v16.6
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
    pub fn new() -> Self { /* ... */ }

    pub fn tick(&mut self, delta_seconds: f32) { /* existing */ }

    // ==================== v16.6: Utilitarian Scoring ====================

    fn calculate_utilitarian_score(&self, action: &Action, entity_id: u64) -> (f32, f32) {
        let mut utility: f32 = 0.0;
        let mut affected_count: f32 = 1.0;

        if let Some(state) = self.entity_states.get(&entity_id) {
            // Base utility from emotional state and contribution
            utility += state.emotional_state.valence * 0.8;
            utility += state.contribution_score * 0.1;
            utility += (state.harmony - 1.0) * 0.5;

            // Estimate affected entities via recent interactions
            affected_count = (state.recent_interactions.len() as f32).max(1.0);

            // Action-specific utility modifiers
            match action {
                Action::Diplomacy { .. } => utility += 1.2,
                Action::Teach { .. } => utility += 1.5,
                Action::Create { .. } => utility += 1.0,
                Action::Harvest { .. } => utility += 0.4,
                Action::ConsultCouncil { .. } => utility += 0.8,
                _ => utility += 0.3,
            }

            // High-valence NPCs generate more positive utility
            if state.emotional_state.valence > 0.6 {
                utility += 0.6;
            }
        }

        let net_utility = utility / affected_count.sqrt(); // Diminishing returns on scale
        (net_utility.clamp(-2.0, 3.0), affected_count)
    }

    pub fn evaluate_moral_reasoning(&self, action: &Action, entity_id: u64) -> MoralEvaluation {
        let mut results = Vec::new();

        // 7 Living Mercy Gates evaluation (existing logic)
        // ... (same as v16.5)

        let (utilitarian_score, affected) = self.calculate_utilitarian_score(action, entity_id);

        let overall = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;

        let primary_justification = if overall > 0.82 && utilitarian_score > 1.0 {
            "Strong Mercy alignment with high estimated net utility".to_string()
        } else if overall > 0.75 {
            "Good Mercy alignment with moderate utility".to_string()
        } else {
            "Requires refinement on Mercy principles or utility impact".to_string()
        };

        MoralEvaluation {
            overall_score: overall,
            gate_results: results,
            primary_justification,
            utilitarian_score,
            net_utility_estimate: utilitarian_score,
        }
    }

    fn decide_action_with_mercy_and_councils(&self, entity_id: u64, action: Action) -> ApprovedAction {
        let moral_eval = self.evaluate_moral_reasoning(&action, entity_id);

        // Combine Mercy score with Utilitarian score for final decision
        let combined_score = (moral_eval.overall_score * 0.7) + (moral_eval.utilitarian_score * 0.3);

        if combined_score < 0.6 {
            return ApprovedAction::Block {
                reason: moral_eval.primary_justification.clone(),
                mercy_lesson: "Insufficient alignment on Mercy principles or net utility".to_string(),
            };
        }

        let council = self.deliberate_with_patsagi_councils(entity_id, &action);

        if combined_score > 0.82 {
            ApprovedAction::Execute(action)
        } else if combined_score > 0.68 {
            ApprovedAction::Transform {
                original: action,
                reason: moral_eval.primary_justification.clone(),
                educational_feedback: council.reward_guidance.clone(),
            }
        } else {
            ApprovedAction::Block {
                reason: moral_eval.primary_justification.clone(),
                mercy_lesson: council.reward_guidance,
            }
        }
    }

    pub fn get_moral_evaluation(&self, entity_id: u64, action: &Action) -> MoralEvaluation {
        self.evaluate_moral_reasoning(action, entity_id)
    }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡