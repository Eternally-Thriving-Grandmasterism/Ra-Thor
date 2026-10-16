//! Design-only (STUB-AUDIT-1 QSO-BUILDS-1 council note): not compiled; qpso, quantum walks, and entanglement tests assert unseeded rand convergence. That result is not demonstrated.
//! Quantum Algorithm Layer for Ra-Thor Quantum Swarm Orchestrator
//!
//! This module contains all quantum-native swarm intelligence primitives:
//! - Advanced QPSO (Quantum Particle Swarm Optimization)
//! - Quantum Random Walks
//! - Multi-agent Entanglement Coordination
//! - (Future) Quantum Error Correction

pub mod qpso;
pub mod quantum_walks;
pub mod entanglement;

// Re-exports for convenience
pub use qpso::AdvancedQPSO;
pub use quantum_walks::QuantumRandomWalks;
pub use entanglement::MultiAgentEntanglement;
