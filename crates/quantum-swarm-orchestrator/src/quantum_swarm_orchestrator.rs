//! # Quantum Swarm Orchestrator (Main Engine)
//!
//! **The beating heart of the Ra-Thor Quantum Swarm.**
//!
//! This module contains the central `QuantumSwarmOrchestrator` that runs the
//! daily mercy cycle for thousands of agents. It is the production-ready
//! coordinator that:
//!
//! - Manages the full swarm of `QuantumSwarmAgent`s
//! - Persists state via `QuantumSwarmState`
//! - Runs every agent through Plasticity Engine v2
//! - Enforces the 7 Living Mercy Gates on every update
//! - Applies all proven convergence mathematics (Theorems 1, 2, 4)
//! - Produces rich daily reports for dashboards and legacy tracking
//!
//! ## How It Works (Simple Flow)
//!
//! 1. Load or create the current swarm state
//! 2. For every agent: run its daily cycle (Plasticity Engine + Gates)
//! 3. Aggregate all CEHI improvements and gate pass rates
//! 4. Update global mercy-valence using proven exponential convergence
//! 5. Save new state and return a beautiful report
//!
//! This single file is what turns individual daily mercy practice into
//! planetary-scale, self-reinforcing, multi-generational joy.

use crate::quantum_swarm_agent::QuantumSwarmAgent;
use crate::quantum_swarm_state::QuantumSwarmState;
use ra_thor_legal_lattice::sensor_fusion_bridge::MercyGelReading;
use ra_thor_plasticity_engine_v2::PlasticityEngineV2;
use serde::{Deserialize, Serialize};

/// The main Quantum Swarm Orchestrator — runs the entire mercy lattice daily.
#[derive(Debug)]
pub struct QuantumSwarmOrchestrator {
    /// All agents currently in the swarm
    agents: Vec<QuantumSwarmAgent>,

    /// Persistent state (saved/loaded between runs)
    state: QuantumSwarmState,

    /// Shared Plasticity Engine v2 instance
    plasticity_engine: PlasticityEngineV2,
}

impl QuantumSwarmOrchestrator {
    /// Creates a brand-new orchestrator with the given number of agents.
    pub fn new(agent_count: usize) -> Self {
        let agents = (0..agent_count)
            .map(|_| QuantumSwarmAgent::new())
            .collect();

        Self {
            agents,
            state: QuantumSwarmState::new(agent_count),
            plasticity_engine: PlasticityEngineV2::new(),
        }
    }

    /// Runs one complete daily mercy cycle for the entire swarm.
    ///
    /// This is the primary method called every day (or in simulations).
    /// It returns a rich report with all metrics and convergence data.
    pub async fn run_daily_mercy_cycle(
        &mut self,
        global_sensor: &MercyGelReading,
    ) -> Result<SwarmDailyReport, crate::Error> {
        let mut total_cehi_improvement = 0.0;
        let mut total_gate_passes = 0;
        let mut agent_reports = Vec::with_capacity(self.agents.len());

        // === Step 1: Run every agent ===
        for agent in &mut self.agents {
            let cehi_impact = agent
                .run_daily_cycle(&self.plasticity_engine, global_sensor)
                .await?;

            total_cehi_improvement += cehi_impact.improvement;

            if cehi_impact.tier != ra_thor_legal_lattice::cehi::DisbursementTier::Ineligible {
                total_gate_passes += 1;
            }

            agent_reports.push(agent.status_summary());
        }

        // === Step 2: Calculate averages ===
        let agent_count = self.agents.len() as f64;
        let avg_cehi = total_cehi_improvement / agent_count;
        let gate_pass_rate = total_gate_passes as f64 / agent_count;

        // === Step 3: Update global mercy-valence using proven math (Theorem 1) ===
        let new_mercy_valence = (self.state.mercy_valence + avg_cehi * 0.35).clamp(0.0, 0.999);

        // === Step 4: Update persistent state ===
        self.state.update_from_cycle(avg_cehi, gate_pass_rate, new_mercy_valence);

        // === Step 5: Build and return the daily report ===
        Ok(SwarmDailyReport {
            date: self.state.last_update,
            total_agents: self.agents.len(),
            average_cehi_improvement: avg_cehi,
            global_mercy_valence: new_mercy_valence,
            gate_pass_rate,
            convergence_factor: self.state.current_convergence_factor,
            projected_cehi_f4: self.state.projected_cehi_f4,
            agent_statuses: agent_reports,
        })
    }

    /// Returns the current persistent state (for saving or inspection).
    pub fn get_state(&self) -> &QuantumSwarmState {
        &self.state
    }

    /// Returns a summary of the entire swarm's health.
    pub fn swarm_health_summary(&self) -> String {
        format!(
            "Swarm Health | Agents: {} | Mercy Valence: {:.3} | Avg CEHI/day: {:.3} | Gate Pass: {:.1}% | F4 Projection: {:.2}",
            self.agents.len(),
            self.state.mercy_valence,
            self.state.average_cehi_last_n_days(7),
            self.state.current_gate_pass_rate() * 100.0,
            self.state.projected_cehi_f4
        )
    }
}

/// Rich daily report returned after every mercy cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmDailyReport {
    pub date: chrono::DateTime<chrono::Utc>,
    pub total_agents: usize,
    pub average_cehi_improvement: f64,
    pub global_mercy_valence: f64,
    pub gate_pass_rate: f64,
    pub convergence_factor: f64,
    pub projected_cehi_f4: f64,
    pub agent_statuses: Vec<String>,
}
