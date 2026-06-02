//! # PATSAGi Integration for Quantum Swarm Orchestrator
//!
//! Brings the full PATSAGi Council system into the core quantum swarm layer.

use powrush::patsagi_councils::{PATSAGiOrchestrator, CouncilMemory};
use mercy::MercyGateStatus;

pub struct QuantumPATSAGiBridge {
    orchestrator: PATSAGiOrchestrator,
    memory: CouncilMemory,
}

impl QuantumPATSAGiBridge {
    pub fn new() -> Self {
        Self {
            orchestrator: PATSAGiOrchestrator::new(),
            memory: CouncilMemory::new(),
        }
    }

    pub async fn evaluate_with_memory(
        &mut self,
        action: &str,
        context: &str,
        mercy_status: &MercyGateStatus,
    ) -> String {
        let consensus = self.orchestrator
            .run_full_consensus(action, context, mercy_status)
            .await;

        // Persist decision
        self.memory.record_consensus(consensus.clone());

        format!(
            "Quantum PATSAGi Evaluation\n\nConsensus: {}\n\nDecisions: {}\n\nMemory: {} decisions recorded",
            if consensus.overall_approved { "Approved" } else { "Vetoed" },
            consensus.decisions.len(),
            self.memory.decisions.len()
        )
    }

    pub fn save_memory(&self, path: &str) -> std::io::Result<()> {
        self.memory.save_to_file(path)
    }
}
