//! Custom Proprietary Powrush MMORPG Simulation Orchestrator
//!
//! Professional production-grade reactive WASM bridge + lifecycle observers.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use postcard;

// ... existing imports and types ...

/// === Binary Serialization Helpers (Postcard) ===

pub fn serialize_wasm_request<T: Serialize>(req: &T) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec(req)
}

pub fn deserialize_wasm_request<T: for<'de> Deserialize<'de>>(data: &[u8]) -> Result<T, postcard::Error> {
    postcard::from_bytes(data)
}

/// === Extended WASM Request Types ===

#[derive(Serialize, Deserialize, Clone)]
pub struct WasmChatMessage {
    pub entity_id: u64,
    pub message: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WasmAbilityUse {
    pub entity_id: u64,
    pub ability_id: u32,
    pub target_id: u64,
}

/// === Bevy Events for WASM Actions ===

#[derive(Event, Clone)]
pub struct WasmChatSent {
    pub entity_id: u64,
    pub message: String,
}

#[derive(Event, Clone)]
pub struct WasmAbilityUsed {
    pub entity_id: u64,
    pub ability_id: u32,
    pub target_id: u64,
}

/// === Observers ===

pub fn on_wasm_chat_sent(
    trigger: Trigger<WasmChatSent>,
    mut unlock_state: ResMut<ServerUnlockState>,
) {
    // Example: Chat activity slightly boosts council influence (social RBE)
    unlock_state.council_influence_progress =
        (unlock_state.council_influence_progress + 0.001).min(1.0);
}

pub fn on_wasm_ability_used(
    trigger: Trigger<WasmAbilityUsed>,
    mut entities: Query<&mut SovereignEntity>,
    mut healing_field: ResMut<CliffordHealingField>,
) {
    let event = trigger.event();

    if let Ok(mut entity) = entities.get_mut(Entity::from_raw(event.entity_id)) {
        // Example effect: ability use gives small contribution to a combat skill
        entity.contribute("combat", 2.0);
        entity.apply_valence_update(0.01);
    }
}

/// === Lifecycle Observers (OnAdd / OnRemove) ===

pub fn on_sovereign_entity_added(
    trigger: Trigger<OnAdd, SovereignEntity>,
    query: Query<&SovereignEntity>,
) {
    if let Ok(entity) = query.get(trigger.entity()) {
        // Automatic behavior on entity spawn
        // Could apply initial blessing, notify WASM clients, etc.
    }
}

pub fn on_sovereign_entity_removed(
    trigger: Trigger<OnRemove, SovereignEntity>,
) {
    // Cleanup logic when a SovereignEntity is despawned
    // Could remove from healing field, update PATSAGi stats, etc.
}

// Update process_wasm_event_queue to handle new types
pub fn process_wasm_event_queue(
    mut queue: ResMut<WasmEventQueue>,
    mut login_writer: EventWriter<EntityLoggedIn>,
    mut contribution_writer: EventWriter<WasmContributionMade>,
    mut chat_writer: EventWriter<WasmChatSent>,
    mut ability_writer: EventWriter<WasmAbilityUsed>,
) {
    // Existing login + contribution handling...

    for chat in queue.pending_chats.drain(..) {
        chat_writer.send(WasmChatSent {
            entity_id: chat.entity_id,
            message: chat.message,
        });
    }

    for ability in queue.pending_abilities.drain(..) {
        ability_writer.send(WasmAbilityUsed {
            entity_id: ability.entity_id,
            ability_id: ability.ability_id,
            target_id: ability.target_id,
        });
    }
}

// Plugin registration
impl Plugin for PowrushSimulationOrchestratorPlugin {
    fn build(&self, app: &mut App) {
        // ... existing ...
        app.add_event::<WasmChatSent>();
        app.add_event::<WasmAbilityUsed>();

        app.add_observer(on_wasm_chat_sent);
        app.add_observer(on_wasm_ability_used);
        app.add_observer(on_sovereign_entity_added);
        app.add_observer(on_sovereign_entity_removed);

        app.add_systems(Update, process_wasm_event_queue);
    }
}

// === Hybrid Native Simulation Shard Planning (Initial Design) ===
// 
// For larger worlds we recommend a hybrid model:
// - WASM shards: Small to medium instances, browser/VR/AR clients, sovereign offline play
// - Native shards: Large-scale persistent worlds, heavy AI/AGI population, high-frequency simulation
// 
// Communication between shards could use a lightweight protocol (postcard over QUIC or custom).
// Each shard would run its own Bevy World + PATSAGi Council instance.
// SovereignEntity migration between shards would be handled via events + serialization.
// 
// This preserves the mercy-gated, RBE-native architecture while scaling globally.
