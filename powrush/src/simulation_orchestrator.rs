//! Cross-Shard Communication System for Powrush
//!
//! Enables shards (WASM and Native) to exchange messages, including entity migrations,
//! chat, ability use, and state synchronization.
//! Currently implemented as an in-memory system for development.
//! Designed to be replaced with real networking (QUIC, WebSocket, or custom binary protocol using postcard).

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::simulation_orchestrator::SovereignEntityMigration;

/// Messages that can be sent between shards
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum CrossShardMessage {
    EntityMigration(SovereignEntityMigration),
    ChatMessage {
        from_entity_id: u64,
        message: String,
    },
    AbilityUse {
        from_entity_id: u64,
        ability_id: u32,
        target_id: u64,
    },
    StateSync {
        // Future: compact world state delta
        shard_id: u64,
        timestamp: u64,
    },
}

/// Resource that acts as the cross-shard communication inbox/outbox
#[derive(Resource, Default)]
pub struct CrossShardCommunicator {
    pub outgoing: VecDeque<CrossShardMessage>,
    pub incoming: VecDeque<CrossShardMessage>,
}

/// Send a message to another shard (currently enqueues for local processing)
pub fn send_cross_shard_message(
    communicator: &mut CrossShardCommunicator,
    message: CrossShardMessage,
) {
    communicator.outgoing.push_back(message);
}

/// Process incoming cross-shard messages
/// This system should run every frame on each shard.
pub fn process_cross_shard_messages(
    mut communicator: ResMut<CrossShardCommunicator>,
    mut commands: Commands,
    mut healing_field: ResMut<crate::clifford_healing_fields::CliffordHealingField>,
    mut unlock_state: ResMut<crate::resources::server_unlock_state::ServerUnlockState>,
) {
    while let Some(message) = communicator.incoming.pop_front() {
        match message {
            CrossShardMessage::EntityMigration(migration) => {
                // Apply the migrated entity on this shard
                let _ = crate::simulation_orchestrator::apply_migrated_entity(
                    &postcard::to_allocvec(&migration).unwrap(), // re-serialize for the function
                    &mut commands,
                    &mut healing_field,
                    &mut unlock_state,
                );
            }
            CrossShardMessage::ChatMessage { from_entity_id, message } => {
                // Could trigger a WasmChatSent event or handle locally
                println!("[CrossShard] Chat from {}: {}", from_entity_id, message);
            }
            CrossShardMessage::AbilityUse { from_entity_id, ability_id, target_id } => {
                println!("[CrossShard] Ability {} used by {} on {}", ability_id, from_entity_id, target_id);
            }
            CrossShardMessage::StateSync { shard_id, timestamp } => {
                println!("[CrossShard] State sync from shard {} at {}", shard_id, timestamp);
            }
        }
    }

    // In a real multi-shard setup, outgoing messages would be sent over the network here
    // and incoming would be populated from network receive.
    // For now, we simulate by moving outgoing to incoming (loopback for testing).
    while let Some(msg) = communicator.outgoing.pop_front() {
        communicator.incoming.push_back(msg);
    }
}

/// Helper to request migration of an entity to another shard
pub fn request_entity_migration(
    entity: &crate::entity::SovereignEntity,
    target_shard_id: u64,
    communicator: &mut CrossShardCommunicator,
) -> Result<(), postcard::Error> {
    let migration_packet = crate::simulation_orchestrator::prepare_entity_for_migration(
        entity,
        0, // source shard (placeholder)
        target_shard_id,
    )?;

    // For now we embed the migration directly. In real use we would deserialize on target.
    let migration = postcard::from_bytes::<SovereignEntityMigration>(&migration_packet)?;

    send_cross_shard_message(
        communicator,
        CrossShardMessage::EntityMigration(migration),
    );

    Ok(())
}
