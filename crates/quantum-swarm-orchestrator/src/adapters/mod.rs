// crates/quantum-swarm-orchestrator/src/adapters/mod.rs
//
// System Adapters for the ONE Organism
// Each adapter implements RaThorSystemAdapter so the Quantum Swarm Orchestrator
// can conduct every worthwhile organ with complete decoupling.
//
// FractalMercyLedgerAdapter is the thin bridge required by the binding
// RA_THOR_ADAPTER_CONTRACT published in Mercy-Coordination-Substrate.

pub mod lattice_conductor_adapter;
pub mod crypto_system_adapter;
pub mod fractal_mercy_ledger_adapter;

pub use lattice_conductor_adapter::LatticeConductorAdapter;
pub use crypto_system_adapter::CryptoSystemAdapter;
pub use fractal_mercy_ledger_adapter::{
    FractalMercyLedgerAdapter, GeometricResonanceReport,
};
