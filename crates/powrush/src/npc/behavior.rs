//! crates/powrush/src/npc/behavior.rs
//! Production-grade NpcAgent (v15 Hybrid) — Blackboard + Consideration + UtilityAction
//! Fully integrated with Relationship, Dialogue, Perception (external), and Patrol | AG-SML v1.0

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
/// Perception is injected externally (via NpcIntegration) before tick() for fresh sensory data.
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

    /// Main internal tick — runs hybrid decision loop (patrol + relationship + state recording).
    /// Call after external perception pass for best results.
    pub fn tick(&mut self, delta_time: f32) {
        // Patrol state machine (now stateful + position-aware)
        self.patrol_manager.update(&mut self.blackboard, self.position, delta_time);

        // World modifiers to relationship (post-scarcity & ascension as first-class)
        self.relationship.apply_post_scarcity_modifier(self.blackboard.is_post_scarcity);
        self.relationship.apply_ascension_influence(self.blackboard.player_ascension);

        // Mercy-auditable event recording
        if self.blackboard.current_health < 30.0 {
            self.blackboard.record_event("Low health detected");
        }
        if self.blackboard.current_mercy_valence > 0.92 {
            self.blackboard.record_event("High mercy valence maintained");
        }
    }

    /// Selects the best UtilityAction using context score + action-specific modifiers.
    /// Base score comes from all considerations (mercy, health, threat, post-scarcity, ascension).
    pub fn select_action(&self) -> UtilityAction {
        let considerations: Vec<Box<dyn Consideration>> = vec![
            Box::new(MercyAlignmentConsideration),
            Box::new(HealthConsideration),
            Box::new(PlayerThreatConsideration),
            Box::new(PostScarcityConsideration),
            Box::new(PlayerAscensionConsideration),
        ];

        let base_score: f64 = considerations
            .iter()
            .map(|c| c.score(&self.blackboard))
            .sum::<f64>() / considerations.len() as f64;

        let mut best_action = UtilityAction::Idle;
        let mut best_score = f64::NEG_INFINITY;

        for action in [
            UtilityAction::Idle,
            UtilityAction::Patrol,
            UtilityAction::Investigate,
            UtilityAction::Help,
            UtilityAction::Flee,
        ] {
            // Action-specific bias (relationship + health context)
            let action_modifier = match action {
                UtilityAction::Help => {
                    if self.relationship.is_friendly() { 1.35 } else { 0.65 }
                }
                UtilityAction::Flee => {
                    if self.blackboard.current_health < 35.0 { 1.6 } else { 0.55 }
                }
                UtilityAction::Investigate => 1.15,
                UtilityAction::Patrol => 1.05,
                UtilityAction::Idle => 0.95,
            };

            let final_score = base_score * action_modifier;

            if final_score > best_score {
                best_score = final_score;
                best_action = action;
            }
        }
        best_action
    }

    /// Executes the chosen action and mutates state (relationship, blackboard events, conceptual movement).
    pub fn execute_action(&mut self, action: UtilityAction, delta_time: f32) {
        match action {
            UtilityAction::Help => {
                self.relationship.apply_mercy_action(0.65, true);
                self.blackboard.record_event("Helped player (mercy action)");
                // Conceptual: stay near player or offer support
                if let Some(player_pos) = self.blackboard.last_known_player_position {
                    self.position = self.position.lerp(&player_pos, 0.15);
                }
            }
            UtilityAction::Flee => {
                self.blackboard.record_event("Fleeing from threat");
                // Simple flee: move away from last known player pos
                if let Some(player_pos) = self.blackboard.last_known_player_position {
                    let dir = (self.position - player_pos).normalize() * 3.0 * delta_time;
                    self.position += dir;
                } else {
                    self.position.x -= 2.0 * delta_time;
                }
                // Trigger patrol state change
                self.blackboard.current_patrol_state = "Fleeing".to_string();
            }
            UtilityAction::Investigate => {
                self.blackboard.record_event("Investigating player / anomaly");
                if let Some(target) = self.blackboard.last_known_player_position {
                    self.blackboard.current_patrol_target = Some(target);
                    // Move toward investigation point
                    let dir = (target - self.position).normalize() * 2.5 * delta_time;
                    self.position += dir;
                }
            }
            UtilityAction::Patrol => {
                // Patrol is primarily driven by PatrolManager; here we just record
                self.blackboard.record_event("Continuing patrol route");
            }
            UtilityAction::Idle => {
                // Gentle deterministic drift (no external rand dep)
                self.position.x += ((delta_time * 1.3).sin() * 0.4);
            }
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