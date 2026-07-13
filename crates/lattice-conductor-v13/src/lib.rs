pub mod self_evolution_telemetry;
pub mod prelude;

// =============================================================================
// Clean Public API Re-exports
// =============================================================================
// For a clean, ergonomic API, all key public types are re-exported directly
// at the crate root. Users can simply do:
//
//   use lattice_conductor_v13::{ConductorSelfEvolutionRecorder, ...};
//   use lattice_conductor_v13::prelude::*;   // even more convenient wildcard import
//
// without needing to know internal module structure.
// Submodules remain available for advanced use if needed.
// =============================================================================

pub use self_evolution_telemetry::ConductorSelfEvolutionRecorder;

// --- Core traits & types ---
pub use crate::conductable::{Conductable, ConductorRegistry, MercyAligned, SystemBlessing};
pub use crate::coordinator::{AverageInfluenceStrategy, CoordinationStrategy, LeaderFollowerStrategy, MercyWeightedStrategy, MultiConductorSimulation};
pub use crate::geometric::{BasicGeometricMotor, GeometricMotor, GeometricState};
pub use crate::self_evolution::{EpigeneticBlessing, SelfEvolving, SelfEvolutionOrchestrator};