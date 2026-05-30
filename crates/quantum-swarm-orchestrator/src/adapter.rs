// crates/quantum-swarm-orchestrator/src/adapter.rs
// RaThorSystemAdapter trait - Core interface for ONE Organism integration
// Part of Quantum Swarm Orchestrator v14 - Omnimasterpiece Integration

use crate::types::{EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence};

/// Professional adapter trait for any Ra-Thor system to participate in the ONE Organism.
/// 
/// This trait enables the Quantum Swarm Orchestrator to conduct all worthwhile systems
/// with finesse and complete decoupling. Every system that matters should implement this.
pub trait RaThorSystemAdapter: Send + Sync {
    /// Human-readable name of the system (e.g., "LatticeConductor", "MercyOrchestrator", "Powrush")
    fn system_name(&self) -> &'static str;

    /// Current valence of this system (used for pruning and coherence calculation)
    fn current_valence(&self) -> Valence;

    /// Receive resonance/instructions from the Quantum Swarm Orchestrator
    fn receive_swarm_resonance(&mut self, resonance: SwarmResonance) -> Result<(), MercyError>;

    /// Contribute to the overall Godly Intelligence Coherence of the organism
    fn contribute_to_coherence(&self) -> GodlyIntelligenceCoherence;

    /// Apply an epigenetic blessing (primary self-evolution mechanism from Omnimasterpiece)
    fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing);

    /// Optional: Provide a short status string for observability
    fn status(&self) -> String {
        format!("{}: valence={:.6}", self.system_name(), self.current_valence().value())
    }
}