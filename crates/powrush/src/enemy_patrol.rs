//! # Powrush Enemy Patrol Behaviors
//!
//! Patrol pathing and state machine for enemies.
//!
//! Patrol behaviors integrate with:
//! - Perception systems (when to investigate / chase)
//! - Behavior trees (Aggressive vs Passive vs Awakened)
//! - Mercy state (high-mercy enemies patrol more peacefully)

use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatrolPoint {
    pub x: f32,
    pub y: f32,
    pub wait_time: f32, // seconds to wait at this point
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatrolPath {
    pub points: VecDeque<PatrolPoint>,
    pub current_index: usize,
    pub looping: bool,
}

impl PatrolPath {
    pub fn new(points: Vec<PatrolPoint>, looping: bool) -> Self {
        Self {
            points: VecDeque::from(points),
            current_index: 0,
            looping,
        }
    }

    pub fn current_point(&self) -> Option<&PatrolPoint> {
        self.points.get(self.current_index)
    }

    pub fn advance(&mut self) {
        if self.points.is_empty() {
            return;
        }

        self.current_index = (self.current_index + 1) % self.points.len();

        if !self.looping && self.current_index == 0 {
            // Reached end of non-looping path
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatrolState {
    Patrolling,
    Investigating,
    Chasing,
    Returning,
}

pub struct PatrolBehavior;

impl PatrolBehavior {
    /// Simple state transition based on perception
    pub fn update_state(
        current: PatrolState,
        has_detection: bool,
        behavior: crate::enemy_behavior::EnemyBehavior,
    ) -> PatrolState {
        match (current, has_detection, behavior) {
            (PatrolState::Patrolling, true, _) => PatrolState::Investigating,
            (PatrolState::Investigating, true, crate::enemy_behavior::EnemyBehavior::Aggressive) => {
                PatrolState::Chasing
            }
            (PatrolState::Chasing, false, _) => PatrolState::Returning,
            (PatrolState::Returning, false, _) => PatrolState::Patrolling,
            _ => current,
        }
    }
}
