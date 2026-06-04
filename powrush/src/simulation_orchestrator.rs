//! Shard Reputation Scoring System for Powrush
//!
//! Provides a continuous reputation score for each shard, enabling finer-grained
//! decision making in reconciliation, consensus, and trust management.

use bevy::prelude::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Resource, Default)]
pub struct ShardReputationTracker {
    /// shard_id -> reputation score (0.0 to 100.0)
    pub scores: HashMap<u64, f32>,
    pub default_score: f32,
}

impl ShardReputationTracker {
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            default_score: 50.0,
        }
    }

    /// Get current reputation score (clamped 0-100)
    pub fn get_reputation(&self, shard_id: u64) -> f32 {
        *self.scores.get(&shard_id).unwrap_or(&self.default_score)
    }

    /// Convert reputation to a weight suitable for reconciliation/consensus (0.0 - 1.0)
    pub fn get_reputation_weight(&self, shard_id: u64) -> f32 {
        let score = self.get_reputation(shard_id);
        // Map 0-100 to 0.1-0.95 range
        0.1 + (score / 100.0) * 0.85
    }

    /// Update reputation based on positive or negative events
    pub fn update_reputation(&mut self, shard_id: u64, delta: f32) {
        let current = self.get_reputation(shard_id);
        let new_score = (current + delta).clamp(0.0, 100.0);
        self.scores.insert(shard_id, new_score);
    }

    /// Apply reputation change from a successful interaction
    pub fn record_positive_event(&mut self, shard_id: u64, magnitude: f32) {
        self.update_reputation(shard_id, magnitude.abs());
    }

    /// Apply reputation penalty
    pub fn record_negative_event(&mut self, shard_id: u64, severity: f32) {
        self.update_reputation(shard_id, -severity.abs());
    }

    /// Periodic reputation normalization / slow drift toward neutral
    pub fn apply_reputation_drift(&mut self) {
        for (_, score) in self.scores.iter_mut() {
            if *score > 50.0 {
                *score = (*score - 0.1).max(50.0);
            } else if *score < 50.0 {
                *score = (*score + 0.1).min(50.0);
            }
        }
    }
}

/// System for periodic reputation maintenance
pub fn shard_reputation_maintenance_system(
    mut tracker: ResMut<ShardReputationTracker>,
) {
    tracker.apply_reputation_drift();
}

/// Integration helper: Get effective trust weight combining discrete trust + reputation
pub fn get_combined_trust_weight(
    trust_tracker: &crate::simulation_orchestrator::ShardTrustTracker,
    reputation_tracker: &ShardReputationTracker,
    shard_id: u64,
) -> f32 {
    let trust_weight = trust_tracker.get_effective_trust(shard_id);
    let reputation_weight = reputation_tracker.get_reputation_weight(shard_id);

    // Blend the two signals
    (trust_weight * 0.6 + reputation_weight * 0.4).clamp(0.05, 0.95)
}
