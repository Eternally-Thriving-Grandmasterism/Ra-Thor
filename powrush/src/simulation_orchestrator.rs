//! Cross-Shard State Synchronization for Powrush
//!
//! Enables shards to periodically share aggregated world state (global metrics,
//! PATSAGi influence, healing field coherence, etc.).
//! This complements entity migration by keeping high-level state consistent
//! across the hybrid WASM/Native shard network.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::simulation_orchestrator::CrossShardMessage;

/// Aggregated global state that can be synced between shards
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GlobalStateSnapshot {
    pub shard_id: u64,
    pub timestamp: u64,

    // High-level aggregated metrics (can be expanded)
    pub total_contributions: u64,
    pub average_valence: f32,
    pub council_influence: f32,
    pub active_entity_count: u32,
    pub healing_coherence: f32,
}

/// Add StateSync variant to CrossShardMessage if not already present
// (We extend the existing enum conceptually here)

/// Generate a snapshot of current global state from this shard
pub fn generate_global_state_snapshot(
    shard_id: u64,
    total_contributions: u64,
    average_valence: f32,
    council_influence: f32,
    active_entity_count: u32,
    healing_coherence: f32,
) -> GlobalStateSnapshot {
    GlobalStateSnapshot {
        shard_id,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        total_contributions,
        average_valence,
        council_influence,
        active_entity_count,
        healing_coherence,
    }
}

/// Reconcile incoming global state snapshot using trust-weighted logic
pub fn reconcile_global_state(
    local: &mut GlobalStateSnapshot,
    incoming: &GlobalStateSnapshot,
    incoming_trust: f32,
) {
    let trust = incoming_trust.clamp(0.0, 1.0);
    let local_weight = 1.0 - trust;

    // Weighted blend for most metrics
    local.average_valence =
        local.average_valence * local_weight + incoming.average_valence * trust;

    local.council_influence =
        local.council_influence * local_weight + incoming.council_influence * trust;

    local.healing_coherence =
        local.healing_coherence * local_weight + incoming.healing_coherence * trust;

    // Contributions are additive across shards
    local.total_contributions += (incoming.total_contributions as f32 * trust) as u64;

    // Take the higher entity count (more conservative)
    if incoming.active_entity_count > local.active_entity_count {
        local.active_entity_count = incoming.active_entity_count;
    }

    // Update timestamp
    if incoming.timestamp > local.timestamp {
        local.timestamp = incoming.timestamp;
    }
}

/// System that can periodically broadcast state syncs (placeholder for now)
pub fn broadcast_state_sync(
    shard_id: u64,
    communicator: &mut crate::simulation_orchestrator::CrossShardCommunicator,
    snapshot: GlobalStateSnapshot,
) {
    // In a real implementation this would serialize and send via network
    // For now we just demonstrate the pattern
    println!("[StateSync] Broadcasting snapshot from shard {}", shard_id);
}
