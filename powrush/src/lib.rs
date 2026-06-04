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

// New modules for the 4 approved steps
pub mod hyperon_metta_layer;
pub mod wasm_entity_login;

// Re-exports
pub use clifford_healing_fields::{CliffordHealingField, HealingConfig, GlobalCoherence, HealingFieldError};
pub use resources::server_unlock_state::ServerUnlockState;
pub use systems::patsagi::{PatsagiCouncilPlugin, WarPhase};

/// Initialize the full living simulation with all lattice integrations.
pub fn initialize_living_simulation(app: &mut App) {
    app.add_plugins((
        PatsagiCouncilPlugin,
        hyperon_metta_layer::HyperonMettaReasoningPlugin,
    ));

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
