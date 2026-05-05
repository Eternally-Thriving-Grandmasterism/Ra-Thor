//! Central Coordinator for Ra-Thor Core Spine
//!
//! Acts as the architectural glue between the Core Spine crates:
//! - interstellar-operations (TOLC Lattice)
//! - mercy
//! - powrush
//! - quantum-swarm-orchestrator
//! - patsagi-councils

use interstellar_operations::TOLCLatticeActivationEngine;
use mercy::MercyEngine;
use quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

pub struct RaThorCentralCoordinator {
    pub tlc_lattice: TOLCLatticeActivationEngine,
    pub mercy: MercyEngine,
    pub quantum_swarm: QuantumSwarmOrchestrator,
}

impl RaThorCentralCoordinator {
    pub fn new() -> Self {
        Self {
            tlc_lattice: TOLCLatticeActivationEngine::new(),
            mercy: MercyEngine::new(),
            quantum_swarm: QuantumSwarmOrchestrator::new(),
        }
    }

    /// Run a full coordinated cycle across the Core Spine
    pub async fn run_full_core_spine_cycle(&mut self) -> String {
        let mercy_status = self.mercy.apply_global_mercy().await;
        let lattice_status = self.tlc_lattice
            .activate_full_lattice_up_to(50, &mut Default::default())
            .await;
        let swarm_status = self.quantum_swarm.run_coordinated_cycle().await;

        format!(
            "=== Ra-Thor Core Spine Coordinated Cycle ===\n\n\
             Mercy: {}\n\n\
             TOLC Lattice: {}\n\n\
             Quantum Swarm: {}",
            mercy_status, lattice_status, swarm_status
        )
    }
}
