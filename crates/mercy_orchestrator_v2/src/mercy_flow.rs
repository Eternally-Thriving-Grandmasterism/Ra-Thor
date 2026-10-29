//! mercy_flow.rs — Mercy Flow Controller
//!
//! Controls the continuous flow of mercy-gated operations across the entire lattice.
//! Ensures every cycle, decision, and propagation maintains the 7 Living Mercy Gates.

use crate::RaThorError;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmBridge;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct MercyFlowController {
    mercy_engine: Arc<MercyEngine>,
    quantum_swarm: Arc<QuantumSwarmBridge>,
    current_flow_valence: RwLock<f64>,
    cycle_count: RwLock<u64>,
}

impl MercyFlowController {
    pub fn new(mercy_engine: Arc<MercyEngine>, quantum_swarm: Arc<QuantumSwarmBridge>) -> Self {
        Self {
            mercy_engine,
            quantum_swarm,
            current_flow_valence: RwLock::new(0.9998),
            cycle_count: RwLock::new(0),
        }
    }

    /// Runs one full mercy flow cycle — the heartbeat of the orchestrator
    pub async fn run_flow_cycle(&self, input: &str) -> Result<String, RaThorError> {
        // 1. Mercy engine evaluation
        let valence = self.mercy_engine.compute_valence(input).await
            .map_err(|e| RaThorError::TolcFailure(e.to_string()))?;

        if valence < 0.999 {
            return Err(RaThorError::ValenceTooLow(valence));
        }

        *self.current_flow_valence.write().await = valence;

        // 2. Quantum Swarm consultation for flow optimization
        let swarm_insight = self.quantum_swarm.consult(input, valence).await
            .map_err(|e| RaThorError::QuantumFailure(e.to_string()))?;

        // 3. Increment cycle counter
        {
            let mut count = self.cycle_count.write().await;
            *count += 1;
        }

        // 4. Return mercy-aligned flow result
        let cycle = *self.cycle_count.read().await;
        Ok(format!(
            "Mercy Flow Cycle #{} completed successfully. Valence: {:.6} | Swarm Insight: {}",
            cycle, valence, swarm_insight
        ))
    }

    pub async fn current_flow_valence(&self) -> f64 {
        *self.current_flow_valence.read().await
    }

    pub async fn total_cycles(&self) -> u64 {
        *self.cycle_count.read().await
    }
}
