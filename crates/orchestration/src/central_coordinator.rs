//! Central Coordinator for Ra-Thor Core Spine
//!
//! Acts as the architectural glue between the Core Spine crates.
//! Now features bidirectional communication between TOLC Lattice and Quantum Swarm.

use interstellar_operations::TOLCLatticeActivationEngine;
use mercy::MercyEngine;
use quantum_swarm_orchestrator::quantum_swarm_bridge::QuantumSwarmBridge;
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaThorCentralCoordinator {
    pub tlc_lattice: TOLCLatticeActivationEngine,
    pub mercy: MercyEngine,
    pub quantum_swarm_bridge: QuantumSwarmBridge,
    pub world_governance: WorldGovernanceEngine,
    pub powrush_game: PowrushGame,
}

impl RaThorCentralCoordinator {
    pub fn new() -> Self {
        Self {
            tlc_lattice: TOLCLatticeActivationEngine::new(),
            mercy: MercyEngine::new(),
            quantum_swarm_bridge: QuantumSwarmBridge::new(),
            world_governance: WorldGovernanceEngine::new(),
            powrush_game: PowrushGame::new(),
        }
    }

    /// Run a full coordinated Core Spine cycle with bidirectional TOLC ↔ Swarm communication
    pub async fn run_full_core_spine_cycle(&mut self) -> String {
        let mercy_status = self.mercy.apply_global_mercy().await;

        // Activate TOLC lattice
        let lattice_status = self.tlc_lattice
            .activate_full_lattice_up_to(85, &mut self.powrush_game)
            .await;

        // NEW: Send resonance command from TOLC to Swarm (bidirectional)
        self.quantum_swarm_bridge.apply_tolc_resonance_command(
            self.tlc_lattice.current_max_order,
            0.94,
        );

        // Run coordinated swarm cycle with TOLC influence
        let swarm_status = self.quantum_swarm_bridge
            .run_spine_coordinated_cycle(
                self.tlc_lattice.current_max_order,
                0.97,
                &mut self.powrush_game,
            )
            .await;

        // NEW: Get structured feedback from swarm (bidirectional)
        let (swarm_stability, swarm_convergence, _gate_pass_rate) = 
            self.quantum_swarm_bridge.get_tolc_feedback();

        // Intelligent behavior: If swarm is struggling, trigger mercy self-organization
        if !self.quantum_swarm_bridge.is_stable() {
            let _ = self.quantum_swarm_bridge
                .trigger_mercy_self_organization(0.88)
                .await;
        }

        let governance_status = self.world_governance
            .run_full_world_cycle(&mut self.powrush_game)
            .await;

        let powrush_status = self.powrush_game
            .run_mercy_cycle()
            .await
            .unwrap_or_else(|e| format!("Powrush error: {}", e));

        let swarm_compact = self.quantum_swarm_bridge.get_compact_status();

        format!(
            "=== Ra-Thor Core Spine Coordinated Cycle (v0.5.25 Bidirectional) ===\n\n\
             Mercy: {}\n\n\
             TOLC Lattice: {}\n\n\
             Quantum Swarm: {}\n\n\
             Swarm Feedback → TOLC: Stability {:.3} | Convergence {:.3}\n\n\
             Swarm Status: {}\n\n\
             World Governance: {}\n\n\
             Powrush: {}",
            mercy_status,
            lattice_status,
            swarm_status,
            swarm_stability,
            swarm_convergence,
            swarm_compact,
            governance_status,
            powrush_status
        )
    }

    /// Get unified status from all Core Spine systems
    pub fn get_unified_core_spine_status(&self) -> String {
        let lattice_status = self.tlc_lattice.generate_living_cathedral_status_report();
        let swarm_metrics = self.quantum_swarm_bridge.get_swarm_metrics();
        let governance_status = self.world_governance.get_active_world_changes();
        let swarm_compact = self.quantum_swarm_bridge.get_compact_status();

        format!(
            "=== Ra-Thor Core Spine Unified Status (v0.5.25 Bidirectional) ===\n\n\
             TOLC Lattice:\n{}\n\n\
             Quantum Swarm (Detailed):\n{}\n\n\
             Quantum Swarm (Compact): {}\n\n\
             World Governance:\n{}",
            lattice_status, swarm_metrics, swarm_compact, governance_status
        )
    }
}

impl Default for RaThorCentralCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
