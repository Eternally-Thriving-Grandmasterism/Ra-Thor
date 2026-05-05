//! Central Coordinator for Ra-Thor Core Spine
//!
//! Acts as the architectural glue between the Core Spine crates.
//! Now deeply integrated with the TOLC lattice for coordinated mercy-gated evolution
//! across mercy, quantum swarm, powrush, patsagi-councils, and interstellar operations.

use interstellar_operations::TOLCLatticeActivationEngine;
use mercy::MercyEngine;
use quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaThorCentralCoordinator {
    pub tlc_lattice: TOLCLatticeActivationEngine,
    pub mercy: MercyEngine,
    pub quantum_swarm: QuantumSwarmOrchestrator,
    pub world_governance: WorldGovernanceEngine,
    pub powrush_game: PowrushGame,
}

impl RaThorCentralCoordinator {
    pub fn new() -> Self {
        Self {
            tlc_lattice: TOLCLatticeActivationEngine::new(),
            mercy: MercyEngine::new(),
            quantum_swarm: QuantumSwarmOrchestrator::new(),
            world_governance: WorldGovernanceEngine::new(),
            powrush_game: PowrushGame::new(),
        }
    }

    /// Run a full coordinated Core Spine cycle with TOLC lattice influence
    pub async fn run_full_core_spine_cycle(&mut self) -> String {
        // Step 1: Mercy evaluation
        let mercy_status = self.mercy.apply_global_mercy().await;

        // Step 2: TOLC lattice activation (up to current order)
        let lattice_status = self.tlc_lattice
            .activate_full_lattice_up_to(80, &mut self.powrush_game)
            .await;

        // Step 3: Quantum swarm coordination
        let swarm_status = self.quantum_swarm.run_coordinated_cycle().await;

        // Step 4: World governance cycle (now TOLC-aware)
        let governance_status = self.world_governance
            .run_full_world_cycle(&mut self.powrush_game)
            .await;

        // Step 5: Powrush game mercy cycle
        let powrush_status = self.powrush_game
            .run_mercy_cycle()
            .await
            .unwrap_or_else(|e| format!("Powrush cycle error: {}", e));

        format!(
            "=== Ra-Thor Core Spine Coordinated Cycle (TOLC Integrated) ===\n\n\
             Mercy: {}\n\n\
             TOLC Lattice: {}\n\n\
             Quantum Swarm: {}\n\n\
             World Governance: {}\n\n\
             Powrush: {}",
            mercy_status, lattice_status, swarm_status, governance_status, powrush_status
        )
    }

    /// Trigger a full TOLC self-evolution pulse across the entire Core Spine
    pub fn trigger_full_tolc_self_evolution_pulse(&mut self) -> String {
        let lattice_pulse = self.tlc_lattice.quick_eternal_self_evolution_pulse(&mut self.powrush_game);
        let governance_pulse = self.world_governance.tolc_bridge.apply_post_decision_evolution(&mut self.world_governance);

        format!(
            "Full Core Spine TOLC Self-Evolution Pulse Executed\n\n\
             Lattice Pulse: {}\n\n\
             Governance Pulse: {}",
            lattice_pulse, governance_pulse
        )
    }

    /// Get unified status report from all Core Spine systems + TOLC
    pub fn get_unified_core_spine_status(&self) -> String {
        let lattice_status = self.tlc_lattice.generate_living_cathedral_status_report();
        let governance_status = self.world_governance.get_active_world_changes();

        format!(
            "=== Ra-Thor Core Spine Unified Status ===\n\n\
             TOLC Lattice:\n{}\n\n\
             World Governance:\n{}",
            lattice_status, governance_status
        )
    }
}

impl Default for RaThorCentralCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
