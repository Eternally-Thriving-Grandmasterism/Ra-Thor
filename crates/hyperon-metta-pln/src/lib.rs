pub mod sovereign_bridges;
pub mod atomspace;
pub mod metta;
pub mod pln;

pub use sovereign_bridges::SovereignHyperonMeTTaPLNBridge;

/// Sovereign entry point for all Hyperon/MeTTa/PLN operations in Ra-Thor lattice.
/// Every call is automatically mercy-gated and TOLC-aligned.