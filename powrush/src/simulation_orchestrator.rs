//! Reputation Decay Mechanisms for Powrush Shard Reputation
//!
//! Reputation slowly decays over time with inactivity, encouraging ongoing
//! positive participation in the shard network.

use bevy::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::simulation_orchestrator::ShardReputationTracker;

impl ShardReputationTracker {
    /// Apply time-based reputation decay
    pub fn apply_reputation_decay(&mut self, decay_rate_per_day: f32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for (shard_id, score) in self.scores.iter_mut() {
            // Simple linear decay toward neutral for now
            // Can be upgraded to exponential later
            let days_inactive = 1.0; // Placeholder - in real system track last activity

            let decay_amount = decay_rate_per_day * days_inactive;

            if *score > 50.0 {
                *score = (*score - decay_amount).max(50.0);
            } else if *score < 50.0 {
                *score = (*score + decay_amount).min(50.0);
            }
        }
    }

    /// More advanced exponential reputation decay
    pub fn apply_exponential_reputation_decay(&mut self, decay_rate: f32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for score in self.scores.values_mut() {
            // Decay toward 50.0 using exponential approach
            let distance_from_neutral = *score - 50.0;
            let new_distance = distance_from_neutral * (1.0 - decay_rate).exp();
            *score = 50.0 + new_distance;
        }
    }
}

/// Periodic system for reputation decay
pub fn shard_reputation_decay_system(
    mut tracker: ResMut<ShardReputationTracker>,
) {
    // Apply gentle daily decay
    tracker.apply_reputation_decay(0.5); // 0.5 points per day toward neutral
}
