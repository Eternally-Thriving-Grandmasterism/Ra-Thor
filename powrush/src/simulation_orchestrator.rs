//! Trust Recovery Mechanisms for Powrush Shard Trust System
//!
//! Shards can recover trust through consistent positive behavior, successful
//! migrations, state syncs, and participation in the larger lattice.

use bevy::prelude::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::simulation_orchestrator::ShardTrust;

#[derive(Resource, Default)]
pub struct ShardTrustTracker {
    pub trusts: HashMap<u64, (ShardTrust, u64)>,
    pub decay_rate: f32,
    /// Number of successful interactions needed to attempt trust upgrade
    pub interactions_for_upgrade: u32,
    pub interaction_counts: HashMap<u64, u32>,
}

impl ShardTrustTracker {
    pub fn new() -> Self {
        Self {
            trusts: HashMap::new(),
            decay_rate: 0.08,
            interactions_for_upgrade: 5,
            interaction_counts: HashMap::new(),
        }
    }

    pub fn get_effective_trust(&self, shard_id: u64) -> f32 {
        if let Some((trust_level, last_update)) = self.trusts.get(&shard_id) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let hours_inactive = ((now - last_update) as f32) / 3600.0;
            let base_weight = trust_level.weight();

            let decayed = base_weight * (-self.decay_rate * hours_inactive).exp();
            decayed.max(0.05)
        } else {
            0.5
        }
    }

    /// Record a successful interaction. Can gradually recover/upgrade trust.
    pub fn record_successful_interaction(&mut self, shard_id: u64, new_trust: ShardTrust) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let count = self.interaction_counts.entry(shard_id).or_insert(0);
        *count += 1;

        // Check if we should attempt to upgrade trust level
        if *count >= self.interactions_for_upgrade {
            if let Some((current_trust, _)) = self.trusts.get(&shard_id) {
                let upgraded = match current_trust {
                    ShardTrust::Low => ShardTrust::Medium,
                    ShardTrust::Medium => ShardTrust::High,
                    ShardTrust::High => ShardTrust::Sovereign,
                    ShardTrust::Sovereign => ShardTrust::Sovereign,
                };

                if upgraded as u8 > *current_trust as u8 {
                    self.trusts.insert(shard_id, (upgraded, now));
                    *count = 0; // reset counter after upgrade
                }
            }
        } else {
            // Even without upgrade, refresh the timestamp
            if let Some((trust_level, _)) = self.trusts.get(&shard_id) {
                self.trusts.insert(shard_id, (*trust_level, now));
            }
        }
    }

    /// Slow passive recovery (can be called periodically)
    pub fn apply_passive_recovery(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for (shard_id, (trust_level, last_update)) in self.trusts.iter_mut() {
            let hours = ((now - *last_update) as f32) / 3600.0;

            // Very slow recovery even with some inactivity (encourages eventual return)
            if hours > 12.0 && *trust_level as u8 < ShardTrust::High as u8 {
                // Small chance to recover one level after long time
                if (hours as u32 % 72) == 0 { // every ~3 days
                    let recovered = match *trust_level {
                        ShardTrust::Low => ShardTrust::Medium,
                        ShardTrust::Medium => ShardTrust::High,
                        _ => *trust_level,
                    };

                    if recovered as u8 > *trust_level as u8 {
                        *trust_level = recovered;
                        *last_update = now;
                    }
                }
            }
        }
    }
}

pub fn shard_trust_maintenance_system(mut tracker: ResMut<ShardTrustTracker>) {
    tracker.apply_passive_recovery();
}
