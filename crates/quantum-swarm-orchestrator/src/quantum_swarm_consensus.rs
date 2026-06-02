//! # Quantum Swarm Consensus Engine
//!
//! Unified consensus layer that combines:
//! - PATSAGi Council decisions
//! - LongHorizonMemory coherence
//! - LatticeConductor modulation
//! - Byzantine-resistant thresholds

use powrush::patsagi_councils::{PATSAGiOrchestrator, PATSAGiConsensus};
use crate::long_horizon_memory::LongHorizonMemory;
use crate::lattice_conductor::LatticeConductor;
use mercy::MercyGateStatus;

pub struct QuantumSwarmConsensus {
    patsagi: PATSAGiOrchestrator,
    memory: LongHorizonMemory,
    conductor: LatticeConductor,
}

impl QuantumSwarmConsensus {
    pub fn new() -> Self {
        Self {
            patsagi: PATSAGiOrchestrator::new(),
            memory: LongHorizonMemory::new(),
            conductor: LatticeConductor::new(),
        }
    }

    pub async fn reach_consensus(
        &mut self,
        action: &str,
        context: &str,
        mercy_status: &MercyGateStatus,
    ) -> PATSAGiConsensus {
        // 1. Get PATSAGi decision
        let mut consensus = self.patsagi
            .run_full_consensus(action, context, mercy_status)
            .await;

        // 2. Apply coherence weighting
        let coherence = self.memory.coherence();
        consensus.final_weight *= coherence;

        // 3. Apply Lattice Conductor modulation
        consensus.final_weight = self.conductor.modulate(consensus.final_weight, coherence);

        // 4. Record in memory
        self.memory.update(consensus.final_weight as f64);

        consensus
    }
}
