// crates/quantum-swarm-orchestrator/src/adapters/mod.rs
//
// System Adapters for the ONE Organism
// Each adapter implements RaThorSystemAdapter so the Quantum Swarm Orchestrator
// can conduct every worthwhile organ with complete decoupling.

pub mod lattice_conductor_adapter;
pub mod crypto_system_adapter;

pub use lattice_conductor_adapter::LatticeConductorAdapter;
pub use crypto_system_adapter::CryptoSystemAdapter;
