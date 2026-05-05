//! Quantum Swarm Bridge for Core Spine Integration
//!
//! Bidirectional communication between TOLC Lattice and Quantum Swarm.
//! Version 0.5.25 — Now reacts to specific TOLC orders (Mercy Gate resonance).

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

    // ==================== TOLC → SWARM ====================

    pub async fn run_spine_coordinated_cycle(
        &mut self,
        tolc_order: u32,
        mercy_valence: f64,
        game: &mut PowrushGame,
    ) -> String {
        self.swarm.inject_tolc_influence(tolc_order, mercy_valence);

        // NEW: Special behavior when order is a Mercy Gate (divisible by 7)
        if tolc_order % 7 == 0 {
            self.handle_mercy_gate_resonance(tolc_order, game).await;
        }

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

    /// NEW: Special Mercy Gate Resonance Event (triggered on orders divisible by 7)
    async fn handle_mercy_gate_resonance(&mut self, order: u32, game: &mut PowrushGame) {
        // Extra powerful effects on Mercy Gate orders
        let resonance_boost = (order as f64 * 420.0).min(185000.0);
        game.boost_faction_joy(powrush::Faction::HarmonyWeavers, resonance_boost);

        // Stronger epigenetic blessing on Mercy Gates
        game.apply_epigenetic_blessing(12);

        // Tell the swarm to enter a special high-resonance state
        self.swarm.enter_mercy_gate_resonance_state(order);
    }

    pub fn apply_tolc_resonance_command(&mut self, order: u32, intensity: f64) {
        self.swarm.inject_tolc_influence(order, intensity);
        self.swarm.apply_resonance_boost(intensity);
    }

    // ==================== SWARM → TOLC ====================

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

    pub fn get_tolc_feedback(&self) -> (f64, f64, f64) {
        (
            self.swarm.get_stability_score(),
            self.swarm.get_convergence_rate(),
            self.swarm.get_mercy_gate_pass_rate(),
        )
    }

    pub fn get_compact_status(&self) -> String {
        format!(
            "Swarm | Stability: {:.3} | Conv: {:.3} | Gates: {:.1}%",
            self.swarm.get_stability_score(),
            self.swarm.get_convergence_rate(),
            self.swarm.get_mercy_gate_pass_rate() * 100.0
        )
    }

    pub async fn trigger_mercy_self_organization(&mut self, intensity: f64) -> String {
        let result = self.swarm.trigger_mercy_self_organization(intensity).await;
        format!("Quantum Swarm Mercy Self-Organization: {}", result)
    }

    pub fn is_stable(&self) -> bool {
        self.swarm.get_stability_score() > 0.92 && self.swarm.get_convergence_rate() > 0.88
    }
}

impl Default for QuantumSwarmBridge {
    fn default() -> Self {
        Self::new()
    }
}
