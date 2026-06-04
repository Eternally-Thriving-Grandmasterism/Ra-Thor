//! Exponential Trust Decay for Powrush Shards
//!
//! More realistic, continuous trust decay using exponential function.
//! Trust weight decreases faster initially and then approaches an asymptote.

use bevy::prelude::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::simulation_orchestrator::ShardTrust;

#[derive(Resource, Default)]
pub struct ShardTrustTracker {
    /// shard_id -> (trust_level, last_interaction_unix)
    pub trusts: HashMap<u64, (ShardTrust, u64)>,
    /// Controls how fast trust decays (higher = faster decay)
    pub decay_rate: f32, // e.g. 0.08 per hour
}

impl ShardTrustTracker {
    pub fn new() -> Self {
        Self {
            trusts: HashMap::new(),
            decay_rate: 0.08, // Reasonable starting value
        }
    }

    /// Returns the current effective trust weight with exponential decay
    pub fn get_effective_trust(&self, shard_id: u64) -> f32 {
        if let Some((trust_level, last_update)) = self.trusts.get(&shard_id) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let hours_inactive = ((now - last_update) as f32) / 3600.0;

            let base_weight = trust_level.weight();

            // Exponential decay: weight * e^(-rate * time)
            let decayed = base_weight * (-self.decay_rate * hours_inactive).exp();

            // Never let trust go below 5%
            decayed.max(0.05)
        } else {
            0.5 // Unknown shard default
        }
    }

    pub fn record_successful_interaction(&mut self, shard_id: u64, new_trust: ShardTrust) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.trusts.insert(shard_id, (new_trust, now));
    }

    /// Applies a hard downgrade for very long inactivity (complements exponential decay)
    pub fn apply_hard_decay(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for (_, (trust_level, last_update)) in self.trusts.iter_mut() {
            let hours = ((now - *last_update) as f32) / 3600.0;

            if hours > 48.0 {
                // Downgrade after 2+ days of complete inactivity
                *trust_level = match *trust_level {
                    ShardTrust::Sovereign => ShardTrust::High,
                    ShardTrust::High => ShardTrust::Medium,
                    ShardTrust::Medium => ShardTrust::Low,
                    ShardTrust::Low => ShardTrust::Low,
                };
                *last_update = now;
            }
        }
    }
}

/// Periodic system for trust maintenance
pub fn shard_trust_maintenance_system(mut tracker: ResMut<ShardTrustTracker>) {
    tracker.apply_hard_decay();
}
