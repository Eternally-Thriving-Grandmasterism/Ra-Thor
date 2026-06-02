//! crates/powrush/src/npc/patrol.rs
//! Patrol State Machine + Path Following for v15 hybrid NPC AI
//! v1.0

use super::blackboard::NpcBlackboard;
use nalgebra::Vector2;

pub type Position = Vector2<f32>;

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
    pub wait_time: f32,
    pub current_wait: f32,
}

impl PatrolPath {
    pub fn new(points: Vec<Position>) -> Self {
        Self {
            points,
            current_index: 0,
            wait_time: 2.0,
            current_wait: 0.0,
        }
    }

    pub fn advance(&mut self) {
        if self.points.is_empty() { return; }
        self.current_index = (self.current_index + 1) % self.points.len();
        self.current_wait = self.wait_time;
    }

    pub fn current_target(&self) -> Option<Position> {
        self.points.get(self.current_index).copied()
    }
}

pub struct PatrolManager;

impl PatrolManager {
    pub fn update(blackboard: &mut NpcBlackboard, patrol: &mut PatrolPath, dt: f32, player_detected: bool) {
        if player_detected {
            blackboard.current_patrol_state = "Chasing".to_string();
            return;
        }

        match blackboard.current_patrol_state.as_str() {
            "Chasing" => {
                if !player_detected {
                    blackboard.current_patrol_state = "Investigating".to_string();
                }
            }
            "Investigating" => {
                patrol.current_wait -= dt;
                if patrol.current_wait <= 0.0 {
                    blackboard.current_patrol_state = "Returning".to_string();
                }
            }
            "Returning" | _ => {
                if let Some(target) = patrol.current_target() {
                    // Simple movement toward target (placeholder)
                    blackboard.current_patrol_target = Some(target);

                    // If close enough, advance
                    if (blackboard.current_patrol_target.unwrap_or(target) - target).magnitude() < 1.0 {
                        patrol.advance();
                        blackboard.current_patrol_state = "Patrolling".to_string();
                    }
                }
            }
        }
    }
}