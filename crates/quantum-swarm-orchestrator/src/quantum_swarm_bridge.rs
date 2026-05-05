//! Quantum Swarm Bridge for Core Spine Integration
//!
//! Provides a clean, mercy-gated interface for the CentralCoordinator
//! and other Core Spine components to coordinate with the Quantum Swarm.
//! Version 0.5.25 — Expanded with richer methods and tighter TOLC coupling.

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
        self.swarm.inject_tolc_influence(tolc_order, mercy_valence);

        let swarm_result = self.swarm.run_coordinated_cycle().await;

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

    // ==================== NEW EXPANDED METHODS ====================

    /// Run a pure swarm cycle without TOLC influence (for testing / isolation)
    pub async fn run_isolated_swarm_cycle(&mut self) -> String {
        self.swarm.run_coordinated_cycle().await
    }

    /// Push current TOLC lattice status into the swarm (one-way sync)
    pub fn sync_tolc_status(&mut self, tolc_order: u32, mercy_valence: f64) {
        self.swarm.inject_tolc_influence(tolc_order, mercy_valence);
    }

    /// Get a compact summary suitable for CentralCoordinator reports
    pub fn get_compact_status(&self) -> String {
        format!(
            "Swarm | Stability: {:.3} | Conv: {:.3} | Gates: {:.1}%",
            self.swarm.get_stability_score(),
            self.swarm.get_convergence_rate(),
            self.swarm.get_mercy_gate_pass_rate() * 100.0
        )
    }

    /// Trigger a mercy-gated self-organization pulse inside the swarm
    pub async fn trigger_mercy_self_organization(&mut self, intensity: f64) -> String {
        let result = self.swarm.trigger_mercy_self_organization(intensity).await;
        format!("Quantum Swarm Mercy Self-Organization: {}", result)
    }

    /// Check if the swarm is currently in a stable convergent state
    pub fn is_stable(&self) -> bool {
        self.swarm.get_stability_score() > 0.92 && self.swarm.get_convergence_rate() > 0.88
    }
}

impl Default for QuantumSwarmBridge {
    fn default() -> Self {
        Self::new()
    }
}
