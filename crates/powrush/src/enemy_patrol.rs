//! # Powrush Enemy Patrol Behaviors
//!
//! Patrol pathing and state machine for enemies with random variation.
//!
//! Patrol behaviors integrate with perception, behavior trees, and mercy state.

use rand::Rng;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatrolPoint {
    pub x: f32,
    pub y: f32,
    pub wait_time: f32,
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
    }

    /// Add random variation to patrol points and wait times
    pub fn add_random_variation(&mut self, position_variance: f32, wait_variance: f32) {
        let mut rng = rand::thread_rng();

        for point in &mut self.points {
            point.x += rng.gen_range(-position_variance..position_variance);
            point.y += rng.gen_range(-position_variance..position_variance);
            point.wait_time += rng.gen_range(-wait_variance..wait_variance);
            point.wait_time = point.wait_time.max(0.5);
        }
    }

    /// Slightly shuffle patrol order (makes routes feel less robotic)
    pub fn slight_random_shuffle(&mut self, shuffle_chance: f32) {
        if shuffle_chance <= 0.0 {
            return;
        }
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < shuffle_chance {
            let len = self.points.len();
            if len >= 2 {
                let i = rng.gen_range(0..len);
                let j = rng.gen_range(0..len);
                if i != j {
                    self.points.swap(i, j);
                }
            }
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
    pub fn update_state(
        current: PatrolState,
        has_detection: bool,
        behavior: crate::enemy_behavior::EnemyBehavior,
    ) -> PatrolState {
        match (current, has_detection, behavior) {
            (PatrolState::Patrolling, true, _) => PatrolState::Investigating,
            (PatrolState::Investigating, true, crate::enemy_behavior::EnemyBehavior::Aggressive) => PatrolState::Chasing,
            (PatrolState::Chasing, false, _) => PatrolState::Returning,
            (PatrolState::Returning, false, _) => PatrolState::Patrolling,
            _ => current,
        }
    }
}
