//! Quantum Swarm Bridge for Core Spine Integration
//!
//! Provides a clean, mercy-gated interface for the CentralCoordinator
//! to coordinate with the Quantum Swarm Orchestrator.
//! Accepts TOLC lattice status and returns structured metrics.

use crate::QuantumSwarmOrchestrator;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmBridge {
    pub swarm: QuantumSwarmOrchestrator,
}

impl QuantumSwarmBridge {
    pub fn new() -> Self {
        Self {
            swarm: QuantumSwarmOrchestrator::new(),
        }
    }

    /// Run a coordinated cycle that accepts TOLC lattice influence
    pub async fn run_spine_coordinated_cycle(
        &mut self,
        tolc_order: u32,
        mercy_valence: f64,
        game: &mut PowrushGame,
    ) -> String {
        // Inject TOLC influence into the swarm
        self.swarm.inject_tolc_influence(tolc_order, mercy_valence);

        // Run the swarm's internal cycle
        let swarm_result = self.swarm.run_coordinated_cycle().await;

        // Apply mercy-gated effects back into the game
        let joy_boost = (tolc_order as f64 * 180.0) + (mercy_valence * 850.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, joy_boost.min(125000.0));

        format!(
            "Quantum Swarm Coordinated Cycle Complete\n\
             TOLC Order: {} | Mercy Valence: {:.2}\n\
             {}\n\
             Joy Boost Applied: +{:.0}",
            tolc_order, mercy_valence, swarm_result, joy_boost.min(125000.0)
        )
    }

    /// Get current swarm metrics for the CentralCoordinator
    pub fn get_swarm_metrics(&self) -> String {
        format!(
            "Quantum Swarm Metrics:\n\
             Stability: {:.4}\n\
             Convergence: {:.4}\n\
             Mercy Gate Pass Rate: {:.2}%\n\
             Active Agents: {}",
            self.swarm.get_stability_score(),
            self.swarm.get_convergence_rate(),
            self.swarm.get_mercy_gate_pass_rate() * 100.0,
            self.swarm.get_active_agent_count()
        )
    }
}

impl Default for QuantumSwarmBridge {
    fn default() -> Self {
        Self::new()
    }
}
