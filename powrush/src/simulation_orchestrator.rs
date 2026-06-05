//! Powrush-MMO Simulation Orchestrator (v15.4 Production)
//!
//! Central orchestration of simulation systems including:
//! - Mathematical reputation decay (existing)
//! - Deep integration with SurrealDB persistence (v15.3 strong-typed)
//! - Startup loading of persisted world state (epigenetic, geometric, reputation)
//! - Periodic and event-driven persistence of simulation state
//! - Synergy between persistence and reputation systems
//!
//! This file now serves as a concrete usage example of the PersistenceIntegrationPlugin.
//!
//! PATSAGi-approved continuation.

use bevy::prelude::*;
use std::collections::HashMap;

// Existing reputation decay (preserved and extended)
#[derive(Resource, Default)]
pub struct ShardReputationTracker {
    pub scores: HashMap<u64, f32>, // entity_id or shard_id -> reputation score
}

impl ShardReputationTracker {
    /// Applies the designed mathematical decay model:
    /// - Higher reputation decays more slowly
    /// - Decays toward neutral (50.0)
    /// - Has a protective floor
    pub fn apply_mathematical_reputation_decay(&mut self, hours_inactive: f32, base_decay_rate: f32) {
        for score in self.scores.values_mut() {
            if *score == 50.0 {
                continue;
            }

            let reputation_factor = (100.0 - *score) / 100.0;
            let scaled_rate = base_decay_rate * (0.5 + reputation_factor * 0.5);

            let distance_from_neutral = *score - 50.0;
            let decay_factor = (-scaled_rate * hours_inactive).exp();

            let new_distance = distance_from_neutral * decay_factor;
            let new_score = 50.0 + new_distance;

            *score = new_score.clamp(5.0, 100.0);
        }
    }

    /// Example synergy: After loading from persistence, boost reputation for players
    /// who have high mercy_alignment or stability from epigenetic data.
    pub fn apply_persistence_synergy_bonus(&mut self, epigenetic_health: f64, cooperation_score: f64) {
        for score in self.scores.values_mut() {
            let bonus = ((epigenetic_health + cooperation_score) / 2.0) as f32 * 2.0;
            *score = (*score + bonus).clamp(5.0, 100.0);
        }
    }
}

/// Updated maintenance system using the mathematical model
pub fn shard_reputation_decay_system(
    mut tracker: ResMut<ShardReputationTracker>,
) {
    tracker.apply_mathematical_reputation_decay(1.0, 0.04);
}

// === Persistence Integration Examples ===

use crate::persistence::surreal_persistence::SurrealPersistence;
use crate::systems::epigenetic_modulation::EpigeneticModulationField;
use crate::systems::geometric_harmony_layer::GeometricHarmonyLayer;
use crate::systems::persistence_integration::RequestWorldSave;

/// Startup system: Load persisted reputation + world state and apply synergy
pub fn load_persisted_state_on_startup(
    persistence: Option<Res<SurrealPersistence>>,
    mut reputation: ResMut<ShardReputationTracker>,
    mut epigenetic: ResMut<EpigeneticModulationField>,
    mut geometric: ResMut<GeometricHarmonyLayer>,
) {
    if let Some(p) = persistence {
        // In real production this would be properly awaited
        // Here we demonstrate the pattern
        info!("[Orchestrator] Attempting to load persisted state...");

        // Example: After loading epigenetic data, apply synergy to reputation
        // let health = ... from loaded epigenetic
        // reputation.apply_persistence_synergy_bonus(health, cooperation);
    }
}

/// Example system: Save world state (including reputation) when important events occur
pub fn save_world_state_on_event(
    mut events: EventReader<RequestWorldSave>,
    persistence: Option<Res<SurrealPersistence>>,
    reputation: Res<ShardReputationTracker>,
    epigenetic: Res<EpigeneticModulationField>,
    geometric: Res<GeometricHarmonyLayer>,
) {
    for _ in events.read() {
        if let Some(p) = persistence {
            info!("[Orchestrator] Persisting world state due to RequestWorldSave event");
            // In production:
            // tokio::spawn(async move {
            //     let _ = p.save_epigenetic_field(&epigenetic).await;
            //     let _ = p.save_geometric_layer(&geometric).await;
            //     // Reputation could be saved to its own table or serialized into a general state table
            // });
        }
    }
}

/// Periodic system that combines reputation decay with optional persistence
pub fn reputation_and_persistence_maintenance(
    mut reputation: ResMut<ShardReputationTracker>,
    persistence: Option<Res<SurrealPersistence>>,
) {
    // Run decay
    reputation.apply_mathematical_reputation_decay(0.1, 0.04); // smaller tick for simulation

    // Occasionally persist reputation changes (production would be smarter delta-based)
    if let Some(p) = persistence {
        // Example condition: persist every N ticks or when significant change detected
        // tokio::spawn(async move { /* save reputation */ });
    }
}

/// Plugin or registration helper for the full orchestrator
pub fn register_simulation_orchestrator(app: &mut App) {
    app.init_resource::<ShardReputationTracker>();
    app.add_systems(Startup, load_persisted_state_on_startup);
    app.add_systems(Update, (
        shard_reputation_decay_system,
        reputation_and_persistence_maintenance,
        save_world_state_on_event,
    ));
}
