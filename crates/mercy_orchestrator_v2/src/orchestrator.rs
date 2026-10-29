//! orchestrator.rs — Core Mercy Orchestrator v2
//!
//! The central coordination engine that ties together all Ra-Thor systems
//! while enforcing the 7 Living Mercy Gates at every step.

use crate::RaThorError;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmBridge;
use ra_thor_kernel::Kernel;
use ra_thor_orchestration::OrchestrationEngine;
use patsagi_councils::PatsagiCouncil;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct MercyOrchestratorV2 {
    mercy_engine: Arc<MercyEngine>,
    quantum_swarm: Arc<QuantumSwarmBridge>,
    kernel: Arc<Kernel>,
    orchestration_engine: Arc<OrchestrationEngine>,
    councils: Vec<Arc<PatsagiCouncil>>,
    current_valence: RwLock<f64>,
}

impl MercyOrchestratorV2 {
    pub fn new() -> Self {
        Self {
            mercy_engine: Arc::new(MercyEngine::new()),
            quantum_swarm: Arc::new(QuantumSwarmBridge::new()),
            kernel: Arc::new(Kernel::new()),
            orchestration_engine: Arc::new(OrchestrationEngine::new()),
            councils: (0..13).map(|i| Arc::new(PatsagiCouncil::new(i))).collect(),
            current_valence: RwLock::new(0.9998),
        }
    }

    /// Main orchestration entry point — runs a full mercy-gated cycle
    pub async fn orchestrate(&self, prompt: &str) -> Result<String, RaThorError> {
        // 1. Mercy gate pre-check
        let valence = self.mercy_engine.compute_valence(prompt).await
            .map_err(|e| RaThorError::TolcFailure(e.to_string()))?;

        if valence < 0.999 {
            return Err(RaThorError::ValenceTooLow(valence));
        }

        *self.current_valence.write().await = valence;

        // 2. Parallel council consultation
        let council_results = futures::future::join_all(
            self.councils.iter().map(|c| c.process(prompt))
        ).await;

        // 3. Quantum Swarm + Kernel coordination
        let swarm_insight = self.quantum_swarm.consult(prompt, valence).await?;
        let kernel_response = self.kernel.process(prompt).await?;

        // 4. Final orchestration synthesis
        let final_response = self.orchestration_engine.synthesize(
            prompt,
            council_results,
            swarm_insight,
            kernel_response,
        ).await?;

        Ok(final_response)
    }

    pub async fn current_valence(&self) -> f64 {
        *self.current_valence.read().await
    }
}
