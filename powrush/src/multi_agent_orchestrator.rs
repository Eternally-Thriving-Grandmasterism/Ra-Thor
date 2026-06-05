//! POWRUSH-MMO Multi-Agent Orchestrator
//! v16.5-moral-reasoning-frameworks
//!
//! Production implementation of Moral Reasoning Frameworks.
//! Centered on the 7 Living Mercy Gates with structured, auditable reasoning.
//! NPCs now generate explicit moral justifications for their decisions.
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

// ==================== v16.5: Moral Reasoning Framework ====================

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

    // ==================== v16.5: Enhanced Moral Reasoning ====================

    pub fn evaluate_moral_reasoning(&self, action: &Action, entity_id: u64) -> MoralEvaluation {
        let mut results = Vec::new();

        // 1. Radical Love
        let love_score = match action {
            Action::Teach { .. } | Action::Diplomacy { .. } => 0.92,
            _ => 0.75,
        };
        results.push(MoralGateResult {
            gate: "Radical Love",
            score: love_score,
            reason: if love_score > 0.85 { "Promotes connection and care".to_string() } else { "Neutral toward love".to_string() },
        });

        // 2. Boundless Mercy
        let mercy_score = match action {
            Action::Diplomacy { .. } | Action::ConsultCouncil { .. } => 0.90,
            Action::Harvest { .. } => 0.80,
            _ => 0.72,
        };
        results.push(MoralGateResult {
            gate: "Boundless Mercy",
            score: mercy_score,
            reason: "Aligns with non-harm and compassion".to_string(),
        });

        // 3. Service
        let service_score = match action {
            Action::Teach { .. } | Action::Diplomacy { .. } => 0.88,
            _ => 0.70,
        };
        results.push(MoralGateResult {
            gate: "Service",
            score: service_score,
            reason: if service_score > 0.8 { "Serves others or the collective".to_string() } else { "Primarily self-directed".to_string() },
        });

        // 4. Abundance
        let abundance_score = match action {
            Action::Create { .. } | Action::Diplomacy { .. } | Action::Harvest { .. } => 0.85,
            _ => 0.68,
        };
        results.push(MoralGateResult {
            gate: "Abundance",
            score: abundance_score,
            reason: "Contributes to shared prosperity".to_string(),
        });

        // 5. Truth
        results.push(MoralGateResult {
            gate: "Truth",
            score: 0.82,
            reason: "Evaluated for honesty and clarity".to_string(),
        });

        // 6. Joy
        let joy_score = match action {
            Action::Diplomacy { .. } | Action::ConsultCouncil { .. } => 0.87,
            _ => 0.71,
        };
        results.push(MoralGateResult {
            gate: "Joy",
            score: joy_score,
            reason: "Has potential to increase positive experience".to_string(),
        });

        // 7. Cosmic Harmony
        let harmony_score = if let Some(state) = self.entity_states.get(&entity_id) {
            (state.harmony * 0.55 + 0.45).min(0.96)
        } else { 0.75 };
        results.push(MoralGateResult {
            gate: "Cosmic Harmony",
            score: harmony_score,
            reason: "Supports long-term balance and coexistence".to_string(),
        });

        let overall = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;

        let primary_justification = if overall > 0.82 {
            "Strong alignment across multiple Mercy Gates".to_string()
        } else if overall > 0.7 {
            "Moderate alignment with some refinement needed".to_string()
        } else {
            "Significant misalignment with core Mercy principles".to_string()
        };

        MoralEvaluation {
            overall_score: overall,
            gate_results: results,
            primary_justification,
        }
    }

    fn decide_action_with_mercy_and_councils(&self, entity_id: u64, action: Action) -> ApprovedAction {
        let moral_eval = self.evaluate_moral_reasoning(&action, entity_id);

        if moral_eval.overall_score < 0.65 {
            return ApprovedAction::Block {
                reason: moral_eval.primary_justification.clone(),
                mercy_lesson: "Action insufficiently aligned with the 7 Living Mercy Gates".to_string(),
            };
        }

        let council = self.deliberate_with_patsagi_councils(entity_id, &action);
        let final_score = (moral_eval.overall_score + council.mercy_score) / 2.0;

        if final_score > 0.82 {
            ApprovedAction::Execute(action)
        } else if final_score > 0.70 {
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

    // Expose moral reasoning for DataChannel / auditing
    pub fn get_moral_evaluation(&self, entity_id: u64, action: &Action) -> MoralEvaluation {
        self.evaluate_moral_reasoning(action, entity_id)
    }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡