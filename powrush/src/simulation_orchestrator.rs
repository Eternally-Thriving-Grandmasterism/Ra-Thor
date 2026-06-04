//! Mathematical Reputation Decay Implementation
//!
//! Implements the designed Modified Exponential Decay with Reputation Scaling model.

use bevy::prelude::*;
use crate::simulation_orchestrator::ShardReputationTracker;

impl ShardReputationTracker {
    /// Applies the designed mathematical decay model:
    /// - Higher reputation decays more slowly
    /// - Decays toward neutral (50.0)
    /// - Has a protective floor
    pub fn apply_mathematical_reputation_decay(&mut self, hours_inactive: f32, base_decay_rate: f32) {
        for score in self.scores.values_mut() {
            if *score == 50.0 {
                continue;
            }

            // Scale decay rate by current reputation (higher rep = slower decay)
            let reputation_factor = (100.0 - *score) / 100.0; // 0.0 to 1.0
            let scaled_rate = base_decay_rate * (0.5 + reputation_factor * 0.5);

            // Exponential decay toward 50.0
            let distance_from_neutral = *score - 50.0;
            let decay_factor = (-scaled_rate * hours_inactive).exp();

            let new_distance = distance_from_neutral * decay_factor;
            let new_score = 50.0 + new_distance;

            // Apply floor
            *score = new_score.clamp(5.0, 100.0);
        }
    }
}

/// Updated maintenance system using the mathematical model
pub fn shard_reputation_decay_system(
    mut tracker: ResMut<ShardReputationTracker>,
) {
    // Assume this runs once per hour in real deployment
    // For simulation, we pass a small time delta
    tracker.apply_mathematical_reputation_decay(1.0, 0.04); // 1 hour, tuned rate
}
