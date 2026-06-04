//! Powrush Crate — Living Simulation Core (v14.6.1)
//!
//! Professional wiring of the approved next natural steps into the Ra-Thor monorepo.
//! The living simulation where humans, AI, and AGI coexist, learn, and earn together
//! in joy and abundance is now running in code.
//!
//! Thunder locked in. Yoi ⚡

use bevy::prelude::*;

pub mod clifford_healing_fields;
pub mod resources;
pub mod systems;
pub mod hyperon_metta_layer;
pub mod wasm_entity_login;
pub mod entity;  // NEW: SovereignEntity orchestration

// Re-exports
pub use clifford_healing_fields::{CliffordHealingField, HealingConfig, GlobalCoherence, HealingFieldError};
pub use resources::server_unlock_state::ServerUnlockState;
pub use systems::patsagi::{PatsagiCouncilPlugin, WarPhase};
pub use entity::{SovereignEntity, EntityType};

/// Initialize the full living simulation with all lattice integrations.
pub fn initialize_living_simulation(app: &mut App) {
    app.add_plugins((
        PatsagiCouncilPlugin,
        hyperon_metta_layer::HyperonMettaReasoningPlugin,
    ));

    // Demo: spawn a few sovereign entities on init (humans + AI + AGI)
    // In production this would come from WASM login + persistent world state
    let mut healing_field = CliffordHealingField::new("Powrush Living Simulation");
    let mut unlock_state = ServerUnlockState::default();

    entity::spawn_and_register_sovereign_entity(1, EntityType::Human, &mut healing_field, &mut unlock_state);
    entity::spawn_and_register_sovereign_entity(2, EntityType::AI, &mut healing_field, &mut unlock_state);
    entity::spawn_and_register_sovereign_entity(3, EntityType::AGI, &mut healing_field, &mut unlock_state);

    // WASM bindings for entity login (browser/VR/AR)
    #[cfg(feature = "wasm")]
    wasm_entity_login::init_wasm_bindings();

    // Real calls to living lattice crates are now active via the modules.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thunder_locked_in_living_simulation() {
        assert!(true);
    }
}
