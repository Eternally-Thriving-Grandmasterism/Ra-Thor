//! crates/powrush/src/npc/behavior.rs
//! v15.3 — Harmony now influences Utility scoring and action selection

use super::{
    NpcBlackboard, Consideration,
    MercyAlignmentConsideration, HealthConsideration, PlayerThreatConsideration,
    PostScarcityConsideration, PlayerAscensionConsideration,
    Relationship, DialogueSystem, DialogueResponse,
    PatrolManager,
};
use nalgebra::Vector2;

pub type Position = Vector2<f32>;

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

    pub fn tick(&mut self, delta_time: f32) {
        self.patrol_manager.update(&mut self.blackboard, self.position, delta_time);
        self.relationship.apply_post_scarcity_modifier(self.blackboard.is_post_scarcity);
        self.relationship.apply_ascension_influence(self.blackboard.player_ascension);

        if self.blackboard.current_health < 30.0 {
            self.blackboard.record_event("Low health detected");
        }
    }

    pub fn select_action(&self) -> UtilityAction {
        let considerations: Vec<Box<dyn Consideration>> = vec![
            Box::new(MercyAlignmentConsideration),
            Box::new(HealthConsideration),
            Box::new(PlayerThreatConsideration),
            Box::new(PostScarcityConsideration),
            Box::new(PlayerAscensionConsideration),
        ];

        let base_score: f64 = considerations.iter()
            .map(|c| c.score(&self.blackboard)).sum::<f64>() / considerations.len() as f64;

        // === Harmony Influence (new in v15.3) ===
        let harmony_bonus = self.get_geometric_harmony() * 0.25;

        let mut best_action = UtilityAction::Idle;
        let mut best_score = f64::NEG_INFINITY;

        for action in [UtilityAction::Idle, UtilityAction::Patrol, UtilityAction::Investigate, UtilityAction::Help, UtilityAction::Flee] {
            let mut action_modifier = match action {
                UtilityAction::Help => if self.relationship.is_friendly() { 1.4 } else { 0.6 },
                UtilityAction::Flee => if self.blackboard.current_health < 35.0 { 1.55 } else { 0.55 },
                UtilityAction::Investigate => 1.15,
                UtilityAction::Patrol => 1.05,
                UtilityAction::Idle => 0.95,
            };

            // High harmony boosts positive/social actions
            if self.get_geometric_harmony() > 0.75 {
                if matches!(action, UtilityAction::Help | UtilityAction::Patrol) {
                    action_modifier *= 1.2;
                }
            }

            let final_score = (base_score + harmony_bonus) * action_modifier;

            if final_score > best_score {
                best_score = final_score;
                best_action = action;
            }
        }
        best_action
    }

    fn get_geometric_harmony(&self) -> f64 {
        if let Some(crate::npc::BlackboardValue::Float(h)) = self.blackboard.get_dynamic(
            &crate::npc::BlackboardKey::Custom("geometric_harmony".to_string())
        ) {
            *h
        } else {
            0.7 // default
        }
    }

    pub fn execute_action(&mut self, action: UtilityAction, delta_time: f32) {
        match action {
            UtilityAction::Help => {
                self.relationship.apply_mercy_action(0.65, true);
                self.blackboard.record_event("Helped player (harmony boosted)");
            }
            UtilityAction::Flee => { self.blackboard.record_event("Fleeing from threat"); }
            UtilityAction::Investigate => { self.blackboard.record_event("Investigating"); }
            _ => {}
        }
    }

    pub fn get_dialogue_response(&self, situation: &str) -> DialogueResponse {
        DialogueSystem::select_response(&self.blackboard, &self.relationship, situation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilityAction {
    Idle, Patrol, Investigate, Help, Flee,
}
