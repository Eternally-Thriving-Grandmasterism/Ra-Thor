//! Cross-Shard State Sync Enhancements
//!
//! Periodic broadcasting, StateSync message variant, and richer per-shard metrics.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::simulation_orchestrator::{CrossShardMessage, GlobalStateSnapshot};

/// Extended GlobalStateSnapshot with more detailed per-shard metrics
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GlobalStateSnapshot {
    pub shard_id: u64,
    pub timestamp: u64,

    // Global aggregates
    pub total_contributions: u64,
    pub average_valence: f32,
    pub council_influence: f32,
    pub active_entity_count: u32,
    pub healing_coherence: f32,

    // Per-shard detailed metrics
    pub shard_contributions: u64,
    pub shard_active_entities: u32,
    pub shard_average_valence: f32,
    pub shard_council_contribution: f32,
    pub shard_healing_coherence: f32,
}

/// Add StateSync variant to CrossShardMessage
// (Extending the existing enum)
// In practice we would modify the enum definition, but here we show usage.

/// Periodic state sync broadcasting system
#[derive(Resource)]
pub struct StateSyncTimer {
    pub timer: Timer,
}

impl Default for StateSyncTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(30.0, TimerMode::Repeating), // Every 30 seconds
        }
    }
}

/// System that periodically broadcasts global state to other shards
pub fn periodic_state_sync_broadcast(
    time: Res<Time>,
    mut timer: ResMut<StateSyncTimer>,
    mut communicator: ResMut<crate::simulation_orchestrator::CrossShardCommunicator>,
    // In real use these would come from actual queries/resources
    shard_id: Res<crate::simulation_orchestrator::ShardId>, // placeholder
) {
    timer.timer.tick(time.delta());

    if timer.timer.just_finished() {
        let snapshot = crate::simulation_orchestrator::generate_global_state_snapshot(
            shard_id.0,
            0,    // total_contributions (would be queried)
            0.75, // average_valence
            0.65, // council_influence
            42,   // active_entity_count
            0.88, // healing_coherence
        );

        // Send via cross-shard communicator
        crate::simulation_orchestrator::send_cross_shard_message(
            &mut communicator,
            CrossShardMessage::StateSync(snapshot),
        );
    }
}

/// Process StateSync messages in cross-shard communication
pub fn process_state_sync(
    snapshot: &GlobalStateSnapshot,
    tracker: &crate::simulation_orchestrator::ShardTrustTracker,
) {
    let trust = tracker.get_effective_trust(snapshot.shard_id);

    // Reconcile using weighted trust
    // (In a real system we would have a local GlobalStateSnapshot resource)
    println!(
        "[StateSync] Received from shard {} with effective trust {:.2}",
        snapshot.shard_id, trust
    );

    // Here we would call reconcile_global_state(...) on a local snapshot resource
}
