//! Enhanced Slashing Penalty Mechanisms for Powrush
//!
//! Production-grade dynamic slashing integrated with reputation scoring.
//! Slashing now directly impacts both discrete trust and continuous reputation.

use bevy::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::simulation_orchestrator::{ShardTrust, ShardTrustTracker, ShardReputationTracker, SlashingReason};

impl ShardTrustTracker {
    /// Apply slashing with reputation impact
    pub fn apply_slashing_penalty(
        &mut self,
        reputation_tracker: &mut ShardReputationTracker,
        shard_id: u64,
        reason: SlashingReason,
        severity: f32,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let severity = severity.clamp(0.2, 1.0);

        // Record slashing event
        self.slashing_history
            .entry(shard_id)
            .or_default()
            .push((reason, now, severity));

        // Downgrade discrete trust
        if let Some((current_trust, _)) = self.trusts.get_mut(&shard_id) {
            let downgrade_steps = (severity * 2.5) as u8;

            for _ in 0..downgrade_steps {
                *current_trust = match *current_trust {
                    ShardTrust::Sovereign => ShardTrust::High,
                    ShardTrust::High => ShardTrust::Medium,
                    ShardTrust::Medium => ShardTrust::Low,
                    ShardTrust::Low => ShardTrust::Low,
                };
            }
        }

        // Heavily penalize continuous reputation
        let reputation_penalty = severity * 25.0; // up to -25 points
        reputation_tracker.record_negative_event(shard_id, reputation_penalty);
    }

    /// Automatic slashing when low-trust shards dominate consensus
    pub fn auto_slash_for_low_trust_dominance(
        &mut self,
        reputation_tracker: &mut ShardReputationTracker,
        shard_id: u64,
        dominance_ratio: f32,
    ) {
        if dominance_ratio > 0.4 {
            let severity = (dominance_ratio - 0.3).clamp(0.2, 0.9);
            self.apply_slashing_penalty(
                reputation_tracker,
                shard_id,
                SlashingReason::LowTrustDomination,
                severity,
            );
        }
    }
}

/// System that can trigger automatic slashing based on consensus behavior
pub fn automatic_slashing_system(
    mut trust_tracker: ResMut<ShardTrustTracker>,
    mut reputation_tracker: ResMut<ShardReputationTracker>,
    // This would normally receive data from consensus system
) {
    // Placeholder for integration with consensus fault detection
    // Example: if low_trust_dominated was detected, call auto_slash_for_low_trust_dominance
}
