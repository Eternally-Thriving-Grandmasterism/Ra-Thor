//! coordination.rs — Mercy-Gated Coordination Layer
//!
//! Handles parallel coordination between councils, quantum swarm, kernel,
//! and domain systems while enforcing mercy gates at every step.

use crate::RaThorError;
use ra_thor_mercy::MercyGateEvaluator;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmBridge;
use ra_thor_kernel::Kernel;
use patsagi_councils::PatsagiCouncil;
use serde::{Serialize, Deserialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResult {
    pub success: bool,
    pub mercy_valence: f64,
    pub coherence_score: f64,
    pub decision_summary: String,
    pub affected_systems: Vec<String>,
}

pub struct CoordinationEngine {
    mercy_evaluator: MercyGateEvaluator,
    quantum_swarm: Arc<QuantumSwarmBridge>,
    kernel: Arc<Kernel>,
    councils: Vec<Arc<PatsagiCouncil>>,
}

impl CoordinationEngine {
    pub fn new(
        mercy_evaluator: MercyGateEvaluator,
        quantum_swarm: Arc<QuantumSwarmBridge>,
        kernel: Arc<Kernel>,
        councils: Vec<Arc<PatsagiCouncil>>,
    ) -> Self {
        Self {
            mercy_evaluator,
            quantum_swarm,
            kernel,
            councils,
        }
    }

    /// Coordinates a full mercy-gated decision cycle across all systems
    pub async fn coordinate(&self, prompt: &str) -> Result<CoordinationResult, RaThorError> {
        // 1. Mercy gate evaluation
        let valence = self.mercy_evaluator.evaluate(prompt);
        if valence < 0.999 {
            return Err(RaThorError::ValenceTooLow(valence));
        }

        // 2. Parallel council input
        let council_inputs = futures::future::join_all(
            self.councils.iter().map(|c| c.process(prompt))
        ).await;

        // 3. Quantum Swarm insight
        let swarm_insight = self.quantum_swarm.consult(prompt, valence).await
            .map_err(|e| RaThorError::QuantumFailure(e.to_string()))?;

        // 4. Kernel orchestration
        let kernel_response = self.kernel.process(prompt).await
            .map_err(|e| RaThorError::Unexpected(e.into()))?;

        // 5. Synthesize final coordination result
        let summary = format!(
            "Coordinated decision: {} | Swarm insight: {} | Kernel: {}",
            council_inputs.len(), swarm_insight, kernel_response
        );

        Ok(CoordinationResult {
            success: true,
            mercy_valence: valence,
            coherence_score: valence * 0.98,
            decision_summary: summary,
            affected_systems: vec![
                "PATSAGi Councils".to_string(),
                "Quantum Swarm".to_string(),
                "Kernel".to_string(),
            ],
        })
    }
}
