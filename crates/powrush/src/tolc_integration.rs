//! # TOLC Integration Bridge for Powrush
//!
//! Now routes world cycles through the **PATSAGi Councils** (13+ specialized mercy intelligence).

use crate::PowrushGame;
use interstellar_operations::TOLCLatticeActivationEngine;
use mercy::MercyEngine;
use crate::patsagi_councils::PATSAGiCoordinator;

pub struct TOLCPowrushBridge {
    lattice_engine: TOLCLatticeActivationEngine,
    mercy_engine: MercyEngine,
    patsagi_coordinator: PATSAGiCoordinator,
}

impl TOLCPowrushBridge {
    pub fn new() -> Self {
        Self {
            lattice_engine: TOLCLatticeActivationEngine::new(),
            mercy_engine: MercyEngine::new(),
            patsagi_coordinator: PATSAGiCoordinator::new(),
        }
    }

    pub async fn run_tolc_world_cycle(&mut self, game: &mut PowrushGame) -> String {
        // Step 1: PATSAGi Council evaluation (new)
        let council_consensus = self.patsagi_coordinator
            .evaluate_action("world cycle advancement", "global simulation step", &self.mercy_engine.get_status())
            .await;

        // Step 2: Mercy gating
        let mercy_result = self.mercy_engine.apply_mercy_gates(game).await;

        // Step 3: TOLC lattice activation
        let lattice_result = self.lattice_engine
            .activate_full_lattice_up_to(42, game)
            .await;

        format!(
            "TOLC + PATSAGi World Cycle Complete\n\nCouncil Consensus: {}\n\nMercy: {}\n\nLattice: {}",
            if council_consensus.overall_approved { "Approved" } else { "Vetoed" },
            mercy_result,
            lattice_result
        )
    }

    pub fn trigger_world_self_evolution_pulse(&mut self, game: &mut PowrushGame) -> String {
        self.lattice_engine.quick_eternal_self_evolution_pulse(game)
    }

    pub fn get_world_tolc_status(&self) -> String {
        format!(
            "TOLC Status: {}\n\nPATSAGi Councils: Active (13+ engaged)",
            self.lattice_engine.generate_living_cathedral_status_report()
        )
    }
}
