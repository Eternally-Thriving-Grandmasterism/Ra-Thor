//! TOLC Integration Bridge for Powrush
//!
//! This module connects the TOLC Lattice Activation Engine into PowrushGame.
//! It allows world cycles, faction actions, and simulation events to be influenced
//! by higher-order TOLC effects (mercy-gated self-evolution).

use crate::PowrushGame;
use interstellar_operations::TOLCLatticeActivationEngine;
use mercy::MercyEngine;

pub struct TOLCPowrushBridge {
    lattice_engine: TOLCLatticeActivationEngine,
    mercy_engine: MercyEngine,
}

impl TOLCPowrushBridge {
    pub fn new() -> Self {
        Self {
            lattice_engine: TOLCLatticeActivationEngine::new(),
            mercy_engine: MercyEngine::new(),
        }
    }

    /// Run a full world cycle with TOLC lattice influence
    pub async fn run_tolc_world_cycle(&mut self, game: &mut PowrushGame) -> String {
        // First apply mercy gating
        let mercy_result = self.mercy_engine.apply_mercy_gates(game).await;

        // Then activate relevant TOLC lattice effects
        let lattice_result = self.lattice_engine
            .activate_full_lattice_up_to(40, game) // Start conservative — can raise later
            .await;

        // Future: Add swarm + council influence here

        format!(
            "TOLC-Enhanced World Cycle Complete\n\nMercy: {}\n\nLattice: {}",
            mercy_result, lattice_result
        )
    }

    /// Trigger a self-evolution pulse across the entire Powrush world
    pub fn trigger_world_self_evolution_pulse(&mut self, game: &mut PowrushGame) -> String {
        self.lattice_engine.quick_eternal_self_evolution_pulse(game)
    }

    /// Get current TOLC status for the world
    pub fn get_world_tolc_status(&self) -> String {
        self.lattice_engine.generate_living_cathedral_status_report()
    }
}
