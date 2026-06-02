//! crates/powrush/src/npc/patrol.rs
//! Patrol state machine and path following for NPCs
//! Integrated with blackboard and perception | v1.0 | AG-SML v1.0

use super::blackboard::{NpcBlackboard, Position};
use nalgebra::Vector2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatrolState {
    Patrolling,
    Investigating,
    Chasing,
    Returning,
}

#[derive(Debug, Clone)]
pub struct PatrolPath {
    pub points: Vec<Position>,
    pub current_index: usize,
    pub wait_timer: f32,
}

impl PatrolPath {
    pub fn new(points: Vec<Position>) -> Self {
        Self { points, current_index: 0, wait_timer: 0.0 }
    }

    pub fn advance(&mut self, blackboard: &mut NpcBlackboard, dt: f32) {
        if self.points.is_empty() { return; }

        let target = self.points[self.current_index];
        blackboard.current_patrol_target = Some(target);

        // Simple arrival check (placeholder distance)
        if let Some(last_known) = blackboard.last_known_player_position {
            if (last_known - target).norm() < 2.0 {
                self.current_index = (self.current_index + 1) % self.points.len();
            }
        }
    }
}

pub struct PatrolManager;

impl PatrolManager {
    pub fn update(blackboard: &mut NpcBlackboard, path: &mut PatrolPath, dt: f32, player_detected: bool) {
        if player_detected {
            blackboard.current_patrol_state = "Investigating".to_string();
            // Logic to switch to Chasing would go here in behavior layer
        } else if blackboard.current_patrol_state == "Investigating" {
            // Return to patrolling after some time (simplified)
            blackboard.current_patrol_state = "Patrolling".to_string();
        }

        path.advance(blackboard, dt);
    }
}