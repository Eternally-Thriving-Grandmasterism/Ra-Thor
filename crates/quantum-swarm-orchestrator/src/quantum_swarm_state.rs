//! # Quantum Swarm State
//!
//! **Persistent, serializable state for the Ra-Thor Quantum Swarm Orchestrator.**
//!
//! This module defines the complete persistent state of the mercy-gated quantum swarm.
//! It is designed to be saved/loaded across daily cycles, multi-year simulations,
//! and even multi-generational (F0 → F4+) legacy projections.
//!
//! ## Key Responsibilities
//!
//! - Store real-time mercy-valence and CEHI history
//! - Track 7 Living Mercy Gates pass rates over time
//! - Maintain agent-level states for thousands of parallel agents
//! - Provide convergence metrics and multi-generational projections
//! - Enable seamless restart of long-running 200-year+ mercy legacy simulations
//!
//! ## Integration
//!
//! Fully compatible with:
//! - `QuantumSwarmOrchestrator` (lib.rs)
//! - All Lyapunov proofs (Theorems 1, 2, 4)
//! - Hebbian mathematical models and convergence bounds
//! - Plasticity Engine v2 + Legal Lattice CEHI calculations

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::quantum_swarm_convergence::{
    exponential_swarm_convergence_bound,
    multi_generational_swarm_compound,
};

/// The complete persistent state of the quantum swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSwarmState {
    /// Current global mercy-valence (0.0–1.0)
    pub mercy_valence: f64,

    /// Number of active agents in the swarm
    pub agent_count: usize,

    /// Timestamp of the last daily cycle
    pub last_update: DateTime<Utc>,

    /// Historical CEHI improvement per day (last 365 entries kept)
    pub cehi_history: Vec<f64>,

    /// Historical 7-Gate pass rate (0.0–1.0) per day
    pub gate_pass_history: Vec<f64>,

    /// Current convergence factor (from Theorem 1)
    pub current_convergence_factor: f64,

    /// Multi-generational projection (F4 = 2226 target)
    pub projected_cehi_f4: f64,

    /// Version for future schema migrations
    pub schema_version: u32,
}

impl QuantumSwarmState {
    /// Creates a fresh initial swarm state.
    pub fn new(agent_count: usize) -> Self {
        Self {
            mercy_valence: 0.62,
            agent_count,
            last_update: Utc::now(),
            cehi_history: Vec::with_capacity(365),
            gate_pass_history: Vec::with_capacity(365),
            current_convergence_factor: 1.0,
            projected_cehi_f4: 4.98,
            schema_version: 1,
        }
    }

    /// Updates the state after one daily swarm cycle.
    pub fn update_from_cycle(
        &mut self,
        cehi_improvement: f64,
        gate_pass_rate: f64,
        new_mercy_valence: f64,
    ) {
        self.mercy_valence = new_mercy_valence;
        self.last_update = Utc::now();

        // Rolling 365-day history
        if self.cehi_history.len() >= 365 {
            self.cehi_history.remove(0);
        }
        self.cehi_history.push(cehi_improvement);

        if self.gate_pass_history.len() >= 365 {
            self.gate_pass_history.remove(0);
        }
        self.gate_pass_history.push(gate_pass_rate);

        // Update convergence factor using proven bound (Theorem 1)
        self.current_convergence_factor =
            exponential_swarm_convergence_bound(self.mercy_valence, 1);

        // Update F4 projection (Theorem 3 multi-generational compounding)
        self.projected_cehi_f4 = 4.98 * multi_generational_swarm_compound(4);
    }

    /// Returns the average CEHI improvement over the last N days.
    pub fn average_cehi_last_n_days(&self, n: usize) -> f64 {
        let len = self.cehi_history.len();
        if len == 0 {
            return 0.0;
        }
        let start = if len > n { len - n } else { 0 };
        let sum: f64 = self.cehi_history[start..].iter().sum();
        sum / (len - start) as f64
    }

    /// Returns the current 7-Gate pass rate (7-day rolling average).
    pub fn current_gate_pass_rate(&self) -> f64 {
        if self.gate_pass_history.is_empty() {
            return 0.0;
        }
        let len = self.gate_pass_history.len();
        let start = if len > 7 { len - 7 } else { 0 };
        let sum: f64 = self.gate_pass_history[start..].iter().sum();
        sum / (len - start) as f64
    }

    /// Serializes the entire state to pretty JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Loads state from JSON (for simulation restarts or deployments).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for QuantumSwarmState {
    fn default() -> Self {
        Self::new(1000)
    }
}
