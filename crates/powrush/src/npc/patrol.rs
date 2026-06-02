//! crates/powrush/src/npc/patrol.rs
//! Patrol State Machine + Path Following for v15 hybrid NPC AI
//! Production-grade stateful PatrolManager | Dynamic path + blackboard integration | AG-SML v1.0

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

/// Stateful per-NPC patrol manager.
/// Owns optional path and current state. Integrates perception data from blackboard.
pub struct PatrolManager {
    pub path: Option<PatrolPath>,
    pub state: PatrolState,
}

impl PatrolManager {
    pub fn new() -> Self {
        Self {
            path: None,
            state: PatrolState::Patrolling,
        }
    }

    pub fn with_path(path: PatrolPath) -> Self {
        Self {
            path: Some(path),
            state: PatrolState::Patrolling,
        }
    }

    /// Called every tick from NpcAgent. Uses blackboard sensory (LOS/audio) to drive state.
    pub fn update(&mut self, blackboard: &mut NpcBlackboard, my_position: Position, dt: f32) {
        let player_detected = blackboard.has_line_of_sight || blackboard.audio_strength > 0.4;

        if player_detected {
            self.state = PatrolState::Chasing;
            blackboard.current_patrol_state = "Chasing".to_string();
            blackboard.record_event("Player detected during patrol");
            return;
        }

        if self.state == PatrolState::Chasing && !player_detected {
            self.state = PatrolState::Investigating;
            blackboard.current_patrol_state = "Investigating".to_string();
            if let Some(ref mut p) = self.path {
                p.current_wait = p.wait_time * 1.5;
            }
            return;
        }

        match self.state {
            PatrolState::Investigating => {
                if let Some(ref mut p) = self.path {
                    p.current_wait -= dt;
                    if p.current_wait <= 0.0 {
                        self.state = PatrolState::Returning;
                        blackboard.current_patrol_state = "Returning".to_string();
                    }
                } else {
                    self.state = PatrolState::Patrolling;
                }
            }
            PatrolState::Returning | PatrolState::Patrolling => {
                if let Some(ref mut patrol) = self.path {
                    if let Some(target) = patrol.current_target() {
                        blackboard.current_patrol_target = Some(target);
                        if (my_position - target).magnitude() < 1.5 {
                            patrol.advance();
                            self.state = PatrolState::Patrolling;
                            blackboard.current_patrol_state = "Patrolling".to_string();
                            blackboard.record_event("Patrol waypoint reached");
                        }
                    }
                } else {
                    blackboard.current_patrol_state = "Idle".to_string();
                }
            }
            _ => {}
        }
    }
}

impl Default for PatrolManager {
    fn default() -> Self {
        Self::new()
    }
}