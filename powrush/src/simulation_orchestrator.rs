//! Shard Migration Logic for Powrush Hybrid Simulation
//!
//! Production-grade logic for moving SovereignEntity between WASM and Native shards.
//! Uses postcard for efficient binary serialization.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use postcard;

use crate::entity::SovereignEntity;
use crate::resources::server_unlock_state::ServerUnlockState;
use crate::clifford_healing_fields::CliffordHealingField;

/// Serializable migration packet for a SovereignEntity
#[derive(Serialize, Deserialize, Clone)]
pub struct SovereignEntityMigration {
    pub entity: SovereignEntity,
    pub source_shard_id: u64,
    pub target_shard_id: u64,
    pub migration_timestamp: u64,
}

/// Prepare a SovereignEntity for migration (serializes it into a postcard packet)
pub fn prepare_entity_for_migration(
    entity: &SovereignEntity,
    source_shard_id: u64,
    target_shard_id: u64,
) -> Result<Vec<u8>, postcard::Error> {
    let migration = SovereignEntityMigration {
        entity: entity.clone(),
        source_shard_id,
        target_shard_id,
        migration_timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    postcard::to_allocvec(&migration)
}

/// Apply a migrated entity into the current shard/world
/// This function should be called on the target shard.
pub fn apply_migrated_entity(
    migration_data: &[u8],
    commands: &mut Commands,
    healing_field: &mut CliffordHealingField,
    unlock_state: &mut ServerUnlockState,
) -> Result<SovereignEntity, postcard::Error> {
    let migration: SovereignEntityMigration = postcard::from_bytes(migration_data)?;

    let entity = migration.entity;

    // Spawn the entity in the new shard
    commands.spawn((
        entity.clone(),
        crate::simulation_orchestrator::Active, // assuming marker is public or re-exported
    ));

    // Re-register to the local healing field
    let _ = healing_field.add_organism(
        entity.id,
        nalgebra::Vector3::new(0.8, 0.7, 0.9),
        nalgebra::Vector3::new(0.85, 0.75, 0.8),
        nalgebra::Vector3::new(0.9, 0.85, 0.95),
        entity.valence as f64,
    );

    // Update local PATSAGi state
    unlock_state.council_influence_progress =
        (unlock_state.council_influence_progress + 0.05).min(1.0);

    Ok(entity)
}

/// Event that can trigger migration (useful for future systems)
#[derive(Event)]
pub struct RequestEntityMigration {
    pub entity_id: u64,
    pub target_shard_id: u64,
}
