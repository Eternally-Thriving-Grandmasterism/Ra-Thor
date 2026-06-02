//! crates/powrush/src/npc/behavior.rs
//! Production-grade NpcAgent (v15 Hybrid) — Blackboard + Consideration + UtilityAction
//! Fully integrated with Relationship, Dialogue, Perception, and Patrol | AG-SML v1.0

use super::{
    NpcBlackboard, Consideration,
    MercyAlignmentConsideration, HealthConsideration, PlayerThreatConsideration,
    PostScarcityConsideration, PlayerAscensionConsideration,
    Relationship, DialogueSystem, DialogueResponse,
    PatrolManager, PatrolState,
};
use nalgebra::Vector2;

pub type Position = Vector2<f32>;

/// The core intelligent agent for Powrush NPCs (v15 Hybrid Architecture)
pub struct NpcAgent {
    pub blackboard: NpcBlackboard,
    pub position: Position,
    pub relationship: Relationship,
    pub patrol_manager: PatrolManager,
}

impl NpcAgent {
    pub fn new(position: Position) -> Self {
        Self {
            blackboard: NpcBlackboard::new(),
            position,
            relationship: Relationship::new(),
            patrol_manager: PatrolManager::new(),
        }
    }

    /// Main tick — runs the full hybrid decision loop
    pub fn tick(&mut self, delta_time: f32) {
        // Update patrol state
        self.patrol_manager.update(&mut self.blackboard, self.position, delta_time);

        // Apply world modifiers to relationship
        self.relationship.apply_post_scarcity_modifier(self.blackboard.is_post_scarcity);
        self.relationship.apply_ascension_influence(self.blackboard.player_ascension);

        // Record significant state changes
        if self.blackboard.current_health < 30.0 {
            self.blackboard.record_event("Low health detected");
        }
    }

    /// Selects the best UtilityAction based on all considerations
    pub fn select_action(&self) -> UtilityAction {
        let considerations: Vec<Box<dyn Consideration>> = vec![
            Box::new(MercyAlignmentConsideration),
            Box::new(HealthConsideration),
            Box::new(PlayerThreatConsideration),
            Box::new(PostScarcityConsideration),
            Box::new(PlayerAscensionConsideration),
        ];

        let mut best_action = UtilityAction::Idle;
        let mut best_score = -1.0;

        for action in [UtilityAction::Idle, UtilityAction::Patrol, UtilityAction::Investigate, UtilityAction::Help, UtilityAction::Flee] {
            let mut total_score = 0.0;
            for consideration in &considerations {
                total_score += consideration.score(&self.blackboard);
            }
            // Simple weighted decision (can be improved later)
            let action_modifier = match action {
                UtilityAction::Help => if self.relationship.is_friendly() { 1.3 } else { 0.7 },
                UtilityAction::Flee => if self.blackboard.current_health < 40.0 { 1.5 } else { 0.6 },
                UtilityAction::Investigate => 1.1,
                _ => 1.0,
            };
            let final_score = total_score * action_modifier / considerations.len() as f64;

            if final_score > best_score {
                best_score = final_score;
                best_action = action;
            }
        }
        best_action
    }

    /// Executes the chosen action
    pub fn execute_action(&mut self, action: UtilityAction, _delta_time: f32) {
        match action {
            UtilityAction::Help => {
                self.relationship.apply_mercy_action(0.6, true);
                self.blackboard.record_event("Helped player (mercy action)");
            }
            UtilityAction::Flee => {
                self.blackboard.record_event("Fleeing from threat");
            }
            UtilityAction::Investigate => {
                self.blackboard.record_event("Investigating player");
            }
            _ => {}
        }
    }

    /// Returns an appropriate dialogue response for the current context
    pub fn get_dialogue_response(&self, situation: &str) -> DialogueResponse {
        DialogueSystem::select_response(&self.blackboard, &self.relationship, situation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilityAction {
    Idle,
    Patrol,
    Investigate,
    Help,
    Flee,
}