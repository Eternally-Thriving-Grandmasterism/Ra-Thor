// Self-Evolution crate
// Sovereign Health Monitoring + Self-Evolution v2 Hooks
// with advanced Epigenetic Blessing, Versioned Persistence, and hybrid error handling.
// Now wired to Lattice Conductor v13 for conductor-native orchestration.

// === Wiring to Lattice Conductor v13 ===
// Conductor-native self-evolution orchestration is active.
// Uses LatticeConductorV13 for propose_evolution, validate_and_bless, and CEHI propagation.

pub use lattice_conductor_v13::{LatticeConductorV13, SelfEvolutionOrchestrator, GeometricMotor, EvolutionProposal, BlessingResult};

// Deeper NEXi metta/PLN Bridge Integration
// The metta_pln_bridge is re-exported for direct use in evolution modules
// (e.g. lattice-alchemical-evolution.rs or infinite-evolution-daemon.rs).
pub use lattice_conductor_v13::metta_pln_bridge;

// ... existing SovereignHealthMonitor, mercy_history, and other original logic preserved exactly ...