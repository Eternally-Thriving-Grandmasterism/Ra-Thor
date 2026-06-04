//! Custom Proprietary Powrush MMORPG Simulation Orchestrator
//!
//! ... (abbreviated for this commit; full previous content preserved)

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ... existing markers, events, queue, etc. ...

/// WASM Contribution Request (already defined)
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmContributionRequest {
    pub entity_id: u64,
    pub skill: String,
    pub amount: f32,
}

/// Bevy Event fired when a WASM contribution request is processed
#[derive(Event, Clone)]
pub struct WasmContributionMade {
    pub entity_id: u64,
    pub skill: String,
    pub amount: f32,
}

/// Observer that reacts to WASM contributions.
/// This closes the full reactive loop: WASM → Queue → Event → Observer → RBE effects + PATSAGi influence.
pub fn on_wasm_contribution_made(
    trigger: Trigger<WasmContributionMade>,
    mut entities: Query<&mut SovereignEntity>,
    mut unlock_state: ResMut<ServerUnlockState>,
    mut healing_field: ResMut<CliffordHealingField>,
) {
    let event = trigger.event();

    // Try to find and update the entity
    if let Ok(mut entity) = entities.get_mut(Entity::from_raw(event.entity_id)) {
        // Apply the contribution using existing RBE logic
        entity.contribute(&event.skill, event.amount);

        // Optional: small immediate valence / healing field feedback
        entity.apply_valence_update(0.02);

        // Boost PATSAGi influence (RBE contribution strengthens the councils)
        unlock_state.council_influence_progress =
            (unlock_state.council_influence_progress + 0.003).min(1.0);

        // Could also trigger a light healing field update here
    }
}

// ... existing process_wasm_event_queue updated to also handle contributions ...

pub fn process_wasm_event_queue(
    mut queue: ResMut<WasmEventQueue>,
    mut login_writer: EventWriter<EntityLoggedIn>,
    mut contribution_writer: EventWriter<WasmContributionMade>,
) {
    // Process logins (existing)
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

    // Process contributions from WASM
    for req in queue.pending_contributions.drain(..) {
        contribution_writer.send(WasmContributionMade {
            entity_id: req.entity_id,
            skill: req.skill,
            amount: req.amount,
        });
    }

    // Future: drain chats and abilities
}

// Plugin registration (add the new observer)
impl Plugin for PowrushSimulationOrchestratorPlugin {
    fn build(&self, app: &mut App) {
        // ... existing init ...
        app.add_event::<WasmContributionMade>();
        app.add_observer(on_wasm_contribution_made);
        // ... rest of registration ...
    }
}

// ... rest of file ...
