/*!
# Prelude

Re-exports the most commonly used items from the `lattice_conductor_v13` crate
for a convenient, ergonomic import experience.

## Usage

```rust
use lattice_conductor_v13::prelude::*;

let recorder = ConductorSelfEvolutionRecorder::new();
let conductor = SimpleLatticeConductor::new();
// ...
```

This keeps the public API clean while still allowing direct access to submodules
when needed (`use lattice_conductor_v13::self_evolution_telemetry::...`).
*/

// === New Telemetry Layer ===
pub use crate::self_evolution_telemetry::ConductorSelfEvolutionRecorder;

// === Core Traits & Common Types ===
pub use crate::conductable::{Conductable, ConductorRegistry, MercyAligned, SystemBlessing};
pub use crate::coordinator::{AverageInfluenceStrategy, CoordinationStrategy, LeaderFollowerStrategy, MercyWeightedStrategy, MultiConductorSimulation};
pub use crate::geometric::{BasicGeometricMotor, GeometricMotor, GeometricState};
pub use crate::self_evolution::{EpigeneticBlessing, SelfEvolving, SelfEvolutionOrchestrator};

// === Configuration & Metrics ===
pub use crate::ConductorSymbolicParameters;
pub use crate::AdaptiveParameters;
pub use crate::Metrics;

// === Operation & Voting ===
pub use crate::Operation;
pub use crate::MercyWeightedVote;

// === GPU / PATSAGi Bridge Types (v13.5+) ===
pub use crate::GpuTaskResult;
pub use crate::MercyGpuAudit;
pub use crate::GpuComputePipeline;

// === Main Conductor Type ===
pub use crate::SimpleLatticeConductor;

// === Deliberation ===
pub use crate::SymbolicDeliberation;

// Future items (feature-gated or new) can be added here as the public API stabilizes.
// Examples: ExternalSymbolicInput, SymbolicSelfProposal (under "self-proposal" feature), etc.
