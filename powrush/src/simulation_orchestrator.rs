//! Automatic Trust Decay System for Powrush Shards
//!
//! Shards gradually lose trust over time if they are inactive or inconsistent.
//! This encourages ongoing alignment with the larger Ra-Thor lattice and PATSAGi principles.

use bevy::prelude::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::simulation_orchestrator::ShardTrust;

/// Tracks trust and activity for each shard
#[derive(Resource, Default)]
pub struct ShardTrustTracker {
    /// shard_id -> (current_trust, last_interaction_unix)
    pub trusts: HashMap<u64, (ShardTrust, u64)>,
    pub decay_rate_per_hour: f32, // How much weight is lost per hour of inactivity
}

impl ShardTrustTracker {
    pub fn new() -> Self {
        Self {
            trusts: HashMap::new(),
            decay_rate_per_hour: 0.05, // 5% trust weight loss per hour
        }
    }

    /// Get current effective trust weight for a shard (with decay applied)
    pub fn get_effective_trust(&self, shard_id: u64) -> f32 {
        if let Some((trust_level, last_update)) = self.trusts.get(&shard_id) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let hours_inactive = ((now - last_update) / 3600) as f32;
            let decay = (hours_inactive * self.decay_rate_per_hour).min(0.7); // cap decay

            let base_weight = trust_level.weight();
            (base_weight * (1.0 - decay)).max(0.1) // never go below 10%
        } else {
            0.5 // Default medium trust for unknown shards
        }
    }

    /// Update trust after successful interaction (migration, state sync, etc.)
    pub fn record_successful_interaction(&mut self, shard_id: u64, new_trust: ShardTrust) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.trusts.insert(shard_id, (new_trust, now));
    }

    /// Manually apply decay (can be called from a periodic system)
    pub fn apply_decay(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for (_, (trust_level, last_update)) in self.trusts.iter_mut() {
            let hours_inactive = ((now - *last_update) / 3600) as f32;
            if hours_inactive > 1.0 {
                // Downgrade trust level if significantly inactive
                if hours_inactive > 24.0 && *trust_level as u8 > ShardTrust::Low as u8 {
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
}

/// System that periodically applies trust decay
pub fn shard_trust_decay_system(mut tracker: ResMut<ShardTrustTracker>) {
    tracker.apply_decay();
}
