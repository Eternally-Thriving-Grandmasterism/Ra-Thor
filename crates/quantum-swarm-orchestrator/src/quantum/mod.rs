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
