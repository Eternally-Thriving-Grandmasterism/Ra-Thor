//! Custom Proprietary Powrush MMORPG Simulation Orchestrator
//!
//! This is the heart of Powrush as a fully original Ra-Thor-native MMORPG.
//! No commercial engines, netcode libraries, or economy systems are used or licensed.
//! Everything is derived from Ra-Thor lattice principles under AG-SML.
//!
//! Core custom systems provided:
//! - Authoritative simulation tick (target 20Hz, mercy-gated)
//! - Integration of SovereignEntity, RBE dividends, PATSAGi influence,
//!   Clifford healing fields, and Hyperon/Metta reasoning in one loop.
//! - Hooks for WASM client sync (browser/VR/AR global access)
//! - Future extension points for sovereign shard networking, quantum-swarm
//!   parallel agent execution, and offline PWA persistence.
//!
//! Comparison to latest MMORPGs (first-principles derivation only):
//! - Instead of traditional client prediction + reconciliation (commercial engines):
//!   Lattice-coherent state with non-bypassable mercy validation at every tick.
//! - Instead of cash-shop / loot-box economies: Native RBE contribution →
//!   universal thriving dividend loop with valence weighting.
//! - Instead of scripted NPC AI: Hyperon/Metta symbolic + neural reasoning for
//!   dynamic AGI agents and PATSAGi-governed world events.
//! - Instead of centralized matchmaking: Sovereign entity login via WASM +
//!   PATSAGi council oversight for global, mercy-aligned coexistence.
//! - Instead of heavy engine physics: Clifford geometric healing/interaction fields
//!   for coherent multi-agent "reality".
//!
//! This ensures Powrush works globally as a living, post-scarcity simulation
//! where humans, AI, and AGI learn, contribute, and thrive together.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use crate::entity::{SovereignEntity, EntityType, distribute_universal_thriving_dividends};
use crate::resources::server_unlock_state::ServerUnlockState;
use crate::clifford_healing_fields::CliffordHealingField;
use crate::hyperon_metta_layer;

/// Marker components
#[derive(Component)]
pub struct Active;

#[derive(Component)]
pub struct HumanPlayer;

/// === WASM Event Serialization Layer ===
///
/// Strategy for Powrush:
/// - Events are serializable with serde
/// - JS/WASM can send JSON or binary (via wasm-bindgen)
/// - Bevy receives them via a queue resource or direct EventWriter when possible
/// - Observers react to the deserialized events

/// Serializable version of login request from WASM/JS
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmLoginRequest {
    pub entity_id: u64,
    pub entity_type: u8, // 0=Human, 1=AI, 2=AGI
}

/// Main login event (already serializable via serde on inner types if needed)
#[derive(Event, Clone)]
pub struct EntityLoggedIn {
    pub entity_id: u64,
    pub entity_type: EntityType,
}

/// Observer for login (from previous expansion)
pub fn on_entity_logged_in(
    trigger: Trigger<EntityLoggedIn>,
    mut commands: Commands,
    mut healing_field: ResMut<CliffordHealingField>,
    mut unlock_state: ResMut<ServerUnlockState>,
) {
    let event = trigger.event();

    let sovereign = SovereignEntity::new(event.entity_id, event.entity_type);

    commands.spawn((
        sovereign.clone(),
        Active,
        match event.entity_type {
            EntityType::Human => HumanPlayer,
            _ => HumanPlayer,
        },
    ));

    let _ = healing_field.add_organism(
        event.entity_id,
        nalgebra::Vector3::new(0.8, 0.7, 0.9),
        nalgebra::Vector3::new(0.85, 0.75, 0.8),
        nalgebra::Vector3::new(0.9, 0.85, 0.95),
        sovereign.valence as f64,
    );

    unlock_state.council_influence_progress =
        (unlock_state.council_influence_progress + 0.08).min(1.0);
}

/// Lifecycle observer
pub fn on_sovereign_entity_added(
    trigger: Trigger<OnAdd, SovereignEntity>,
    _query: Query<&SovereignEntity>,
) {
    // React to new SovereignEntity being added
}

/// Contribution event (serializable)
#[derive(Event, Clone)]
pub struct EntityContributionMade {
    pub entity_id: u64,
    pub skill: String,
    pub amount: f32,
}

pub fn on_entity_contribution_made(
    trigger: Trigger<EntityContributionMade>,
    mut unlock_state: ResMut<ServerUnlockState>,
) {
    unlock_state.council_influence_progress =
        (unlock_state.council_influence_progress + 0.005).min(1.0);
}

/// Resource to hold pending events coming from WASM/JS
#[derive(Resource, Default)]
pub struct WasmEventQueue {
    pub pending_logins: Vec<WasmLoginRequest>,
}

/// System that drains the WASM queue and sends proper Bevy events
pub fn process_wasm_event_queue(
    mut queue: ResMut<WasmEventQueue>,
    mut login_writer: EventWriter<EntityLoggedIn>,
) {
    for req in queue.pending_logins.drain(..) {
        let entity_type = match req.entity_type {
            0 => EntityType::Human,
            1 => EntityType::AI,
            2 => EntityType::AGI,
            _ => EntityType::Human,
        };

        login_writer.send(EntityLoggedIn {
            entity_id: req.entity_id,
            entity_type,
        });
    }
}

/// Plugin
pub struct PowrushSimulationOrchestratorPlugin;

impl Plugin for PowrushSimulationOrchestratorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CliffordHealingField>();
        app.init_resource::<WasmEventQueue>();

        app.add_event::<EntityLoggedIn>();
        app.add_event::<EntityContributionMade>();

        app.add_observer(on_entity_logged_in);
        app.add_observer(on_sovereign_entity_added);
        app.add_observer(on_entity_contribution_made);

        app.add_systems(Update, process_wasm_event_queue);
        app.add_systems(Update, powrush_authoritative_tick);
    }
}

fn powrush_authoritative_tick(
    mut changed_entities: Query<&mut SovereignEntity, Changed<SovereignEntity>>,
    mut healing_field: ResMut<CliffordHealingField>,
    mut unlock_state: ResMut<ServerUnlockState>,
    mut contribution_writer: EventWriter<EntityContributionMade>,
) {
    let mut entity_slice: Vec<_> = changed_entities.iter_mut().collect();

    if !entity_slice.is_empty() {
        distribute_universal_thriving_dividends(
            &mut entity_slice,
            &healing_field,
            &unlock_state,
        );

        for entity in &entity_slice {
            contribution_writer.send(EntityContributionMade {
                entity_id: entity.id,
                skill: "coexistence".to_string(),
                amount: 1.0,
            });
        }

        let avg_valence: f32 = entity_slice.iter().map(|e| e.valence).sum::<f32>()
            / entity_slice.len() as f32;
        unlock_state.apply_rbe_thriving_influence(0, avg_valence);
    }
}

/// WASM-exposed helper (can be called from JS via wasm-bindgen in wasm_entity_login.rs)
/// In a full build, this would push into the WasmEventQueue resource.
#[cfg(feature = "wasm")]
pub fn queue_wasm_login_request(entity_id: u64, entity_type: u8) {
    // This is a placeholder showing the serialization bridge.
    // In production: access the Bevy world or use a channel to push to WasmEventQueue.
    // Example: wasm_event_queue.pending_logins.push(WasmLoginRequest { entity_id, entity_type });
}
