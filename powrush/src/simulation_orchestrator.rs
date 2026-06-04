//! Reputation Staking for Powrush Shard Reputation System
//!
//! Allows shards to stake reputation as collateral for high-impact actions
//! (e.g., creating important consensus proposals). Staked reputation can be
//! slashed for bad behavior, creating skin-in-the-game.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::simulation_orchestrator::ShardReputationTracker;

impl ShardReputationTracker {
    /// Stake reputation (locks a portion of reputation score)
    pub fn stake_reputation(&mut self, shard_id: u64, amount: f32) -> bool {
        let current = self.get_reputation(shard_id);

        if amount > current * 0.5 {
            return false; // Cannot stake more than 50% of current reputation
        }

        let staked = self.staked_reputation.entry(shard_id).or_insert(0.0);
        *staked += amount;

        // Reduce available reputation while staked
        self.scores.insert(shard_id, current - amount);
        true
    }

    /// Unstake reputation (releases locked reputation)
    pub fn unstake_reputation(&mut self, shard_id: u64, amount: f32) -> bool {
        let staked = self.staked_reputation.entry(shard_id).or_insert(0.0);

        if amount > *staked {
            return false;
        }

        *staked -= amount;

        // Return reputation to available score
        let current = self.get_reputation(shard_id);
        self.scores.insert(shard_id, current + amount);
        true
    }

    /// Slash staked reputation (used for penalties on high-impact actions)
    pub fn slash_staked_reputation(&mut self, shard_id: u64, amount: f32) {
        let staked = self.staked_reputation.entry(shard_id).or_insert(0.0);
        let slash_amount = amount.min(*staked);

        *staked -= slash_amount;

        // Also reduce overall reputation
        let current = self.get_reputation(shard_id);
        self.scores.insert(shard_id, (current - slash_amount * 0.5).max(0.0));
    }

    /// Get total staked reputation for a shard
    pub fn get_staked_reputation(&self, shard_id: u64) -> f32 {
        *self.staked_reputation.get(&shard_id).unwrap_or(&0.0)
    }

    /// Get available (unstaked) reputation
    pub fn get_available_reputation(&self, shard_id: u64) -> f32 {
        self.get_reputation(shard_id) - self.get_staked_reputation(shard_id)
    }
}

// Extend ShardReputationTracker with staked_reputation field
// (In full implementation, add to struct definition)
// pub staked_reputation: HashMap<u64, f32>,
