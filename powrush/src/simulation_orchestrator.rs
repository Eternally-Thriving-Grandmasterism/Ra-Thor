//! Dynamic Slashing Penalties for Powrush Shard Trust System
//!
//! Penalizes shards that exhibit malicious, inconsistent, or negligent behavior.
//! Penalties are dynamic based on severity, frequency, and current trust level.

use bevy::prelude::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::simulation_orchestrator::ShardTrust;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashingReason {
    LowTrustDomination,
    InconsistentStateSync,
    SpammingProposals,
    VotingAgainstStrongConsensus,
    Other,
}

#[derive(Resource, Default)]
pub struct ShardTrustTracker {
    pub trusts: HashMap<u64, (ShardTrust, u64)>,
    pub decay_rate: f32,
    pub interaction_counts: HashMap<u64, u32>,
    pub slashing_history: HashMap<u64, Vec<(SlashingReason, u64, f32)>>, // (reason, time, severity)
}

impl ShardTrustTracker {
    /// Apply a dynamic slashing penalty
    pub fn apply_slashing_penalty(
        &mut self,
        shard_id: u64,
        reason: SlashingReason,
        severity: f32, // 0.0 - 1.0
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let severity = severity.clamp(0.1, 1.0);

        // Record the slashing event
        self.slashing_history
            .entry(shard_id)
            .or_default()
            .push((reason, now, severity));

        // Apply trust downgrade based on severity and current level
        if let Some((current_trust, last_update)) = self.trusts.get_mut(&shard_id) {
            let downgrade_amount = (severity * 2.0) as u8; // severity scales impact

            let new_trust = match *current_trust {
                ShardTrust::Sovereign if downgrade_amount >= 2 => ShardTrust::High,
                ShardTrust::High if downgrade_amount >= 2 => ShardTrust::Medium,
                ShardTrust::Medium if downgrade_amount >= 1 => ShardTrust::Low,
                ShardTrust::Low => ShardTrust::Low,
                _ => *current_trust,
            };

            if new_trust as u8 < *current_trust as u8 {
                *current_trust = new_trust;
            }

            // Also apply temporary weight penalty
            // (we can extend this later with a separate multiplier)
        }
    }

    /// Gradually lift slashing effects over time (rehabilitation)
    pub fn apply_slashing_rehabilitation(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for (shard_id, history) in self.slashing_history.iter_mut() {
            // Remove old slashing events (older than 7 days)
            history.retain(|(_, timestamp, _)| now - timestamp < 7 * 24 * 3600);

            // If no recent slashing, slowly allow trust recovery
            if history.is_empty() {
                if let Some((trust_level, _)) = self.trusts.get_mut(shard_id) {
                    // Small chance to recover one level if clean for a while
                    if (now % 86400) < 3600 {
                        // once per day window
                        *trust_level = match *trust_level {
                            ShardTrust::Low => ShardTrust::Medium,
                            _ => *trust_level,
                        };
                    }
                }
            }
        }
    }
}

pub fn shard_trust_maintenance_system(mut tracker: ResMut<ShardTrustTracker>) {
    tracker.apply_hard_decay();
    tracker.apply_passive_recovery();
    tracker.apply_slashing_rehabilitation();
}
