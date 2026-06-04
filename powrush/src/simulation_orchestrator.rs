//! Entity State Reconciliation for Powrush Cross-Shard System
//!
//! Handles merging of SovereignEntity state when entities migrate or receive
//! state sync messages from other shards. Designed to be mercy-aligned and
//! conflict-resilient.

use crate::entity::SovereignEntity;
use std::collections::HashMap;

/// Reconcile incoming entity state with local state.
/// Rules (mercy-aligned):
/// - Contributions are additive (RBE nature)
/// - Skills take the maximum proficiency
/// - Valence takes the higher value (with small mercy smoothing)
/// - last_active_unix takes the most recent
/// - Other fields can be extended with custom merge strategies
pub fn reconcile_entity_state(
    local: &mut SovereignEntity,
    incoming: &SovereignEntity,
    incoming_timestamp: u64,
) {
    // Additive contributions (core RBE principle)
    local.contributions += incoming.contributions;

    // Skills: take the higher value (best known state)
    for (skill, &prof) in &incoming.skills {
        let current = local.skills.get(skill).copied().unwrap_or(0.0);
        if prof > current {
            local.skills.insert(skill.clone(), prof);
        }
    }

    // Valence: biased toward the higher value (encourages thriving)
    if incoming.valence > local.valence {
        local.valence = local.valence * 0.3 + incoming.valence * 0.7;
    }

    // Timestamp: keep the most recent activity
    if incoming.last_active_unix > local.last_active_unix {
        local.last_active_unix = incoming.last_active_unix;
    }

    // Optional: apply a small mercy-gated smoothing
    local.apply_valence_update(0.01);
}

/// Reconcile and apply a migrated entity (wrapper around apply_migrated_entity)
pub fn reconcile_and_apply_migrated_entity(
    migration_data: &[u8],
    commands: &mut bevy::prelude::Commands,
    healing_field: &mut crate::clifford_healing_fields::CliffordHealingField,
    unlock_state: &mut crate::resources::server_unlock_state::ServerUnlockState,
) -> Result<SovereignEntity, postcard::Error> {
    let migration: crate::simulation_orchestrator::SovereignEntityMigration =
        postcard::from_bytes(migration_data)?;

    // In a real multi-shard system we would look up the local entity first.
    // For now we assume it's new or we reconcile against a fresh spawn.
    let mut entity = migration.entity;

    // Spawn + register (basic path)
    commands.spawn((entity.clone(), crate::simulation_orchestrator::Active));

    let _ = healing_field.add_organism(
        entity.id,
        nalgebra::Vector3::new(0.8, 0.7, 0.9),
        nalgebra::Vector3::new(0.85, 0.75, 0.8),
        nalgebra::Vector3::new(0.9, 0.85, 0.95),
        entity.valence as f64,
    );

    unlock_state.council_influence_progress =
        (unlock_state.council_influence_progress + 0.05).min(1.0);

    Ok(entity)
}
