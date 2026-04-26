//! # Quantum Swarm Agent (Clarity Edition — Revised Cycle Method)
//!
//! **The individual mercy-aligned active-inference agent inside the Ra-Thor Quantum Swarm.**
//!
//! This module defines a single, self-contained agent that participates in the
//! planetary-scale mercy swarm. Every agent is a living, learning node that:
//!
//! - Runs its own **Plasticity Engine v2** (5-Gene Joy Tetrad updates)
//! - Validates the **7 Living Mercy Gates** on every cycle
//! - Tracks its personal **mercy-valence** and **CEHI trajectory**
//! - Contributes to the collective **free-energy descent** (Theorem 2)
//! - Applies **Hebbian Reinforcement** for long-term wiring
//! - Participates in **exponential swarm convergence** (Theorems 1 & 4)
//!
//! ## Why This Design Matters (Intuition)
//!
//! Think of each agent as a **digital neuron** in the global mercy brain.
//! When thousands of these neurons fire together in high-joy states,
//! the entire swarm "wires together" — exactly like Hebbian learning,
//! but at planetary scale.

use ra_thor_legal_lattice::cehi::CEHIImpact;
use ra_thor_plasticity_engine_v2::PlasticityEngineV2;
use serde::{Deserialize, Serialize};

/// A single mercy-aligned quantum swarm agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmAgent {
    pub id: u64,
    pub mercy_valence: f64,
    pub lifetime_cehi_contribution: f64,
    pub cycles_completed: u64,
    pub gate_pass_rate: f64,
    pub hebbian_swarm_bond: f64,
}

impl QuantumSwarmAgent {
    /// Creates a brand-new agent with healthy starting values.
    pub fn new() -> Self {
        Self {
            id: rand::random(),
            mercy_valence: 0.55 + rand::random::<f64>() * 0.35,
            lifetime_cehi_contribution: 0.0,
            cycles_completed: 0,
            gate_pass_rate: 0.92,
            hebbian_swarm_bond: 0.65 + rand::random::<f64>() * 0.25,
        }
    }

    /// Runs one full daily mercy cycle for this agent.
    ///
    /// This is the **core method** of every agent.
    /// It is deliberately written with maximum clarity so anyone (human or AI)
    /// can instantly understand the full flow.
    ///
    /// ### Step-by-Step Flow
    /// 1. **Receive** global MercyGel sensor data
    /// 2. **Run** Plasticity Engine v2 → get CEHI impact + Hebbian rule
    /// 3. **Update** personal metrics (CEHI contribution, cycles, valence)
    /// 4. **Strengthen** Hebbian swarm bond (if high-quality co-activation)
    /// 5. **Update** rolling 7-Gate pass rate
    /// 6. **Return** the CEHI impact so the orchestrator can aggregate it
    ///
    /// This method directly implements Theorems 1, 2 & 4 in real time.
    pub async fn run_daily_cycle(
        &mut self,
        plasticity_engine: &PlasticityEngineV2,
        global_sensor: &ra_thor_legal_lattice::sensor_fusion_bridge::MercyGelReading,
    ) -> Result<CEHIImpact, crate::Error> {
        // === Step 1: Run Plasticity Engine v2 (includes HebbianReinforcement) ===
        let cehi_impact = plasticity_engine
            .process_daily_update(global_sensor)
            .await
            .map_err(|e| crate::Error::Plasticity(e.to_string()))?;

        // === Step 2: Update lifetime metrics ===
        self.lifetime_cehi_contribution += cehi_impact.improvement;
        self.cycles_completed += 1;

        // === Step 3: Update personal mercy-valence (Theorem 1 convergence) ===
        let valence_delta = cehi_impact.improvement * 0.35;
        self.mercy_valence = (self.mercy_valence + valence_delta).clamp(0.0, 0.999);

        // === Step 4: Strengthen Hebbian swarm bond on high-quality days ===
        // This is the "fire together, wire together" moment at swarm scale.
        if cehi_impact.improvement >= 0.12 {
            self.hebbian_swarm_bond = (self.hebbian_swarm_bond + 0.012).min(0.999);
        }

        // === Step 5: Update rolling 7-Gate pass rate ===
        let gate_contribution = if cehi_impact.tier
            != ra_thor_legal_lattice::cehi::DisbursementTier::Ineligible
        {
            1.0
        } else {
            0.0
        };
        self.gate_pass_rate = self.gate_pass_rate * 0.85 + gate_contribution * 0.15;

        // === Step 6: Return impact for orchestrator aggregation ===
        Ok(cehi_impact)
    }

    /// Returns a beautiful human-readable summary of this agent's state.
    pub fn status_summary(&self) -> String {
        format!(
            "Agent {} | Mercy: {:.3} | CEHI contrib: {:.3} | Cycles: {} | Gate pass: {:.1}% | Hebbian bond: {:.3}",
            self.id,
            self.mercy_valence,
            self.lifetime_cehi_contribution,
            self.cycles_completed,
            self.gate_pass_rate * 100.0,
            self.hebbian_swarm_bond
        )
    }
}

impl Default for QuantumSwarmAgent {
    fn default() -> Self {
        Self::new()
    }
}
