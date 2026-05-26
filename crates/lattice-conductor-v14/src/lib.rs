// crates/lattice-conductor-v14/src/lib.rs
// Lattice Conductor v14 — Central Nervous System of Ra-Thor
// v14.0.5 Thunder Lattice — Distributed Mercy Mesh Foundation added

pub mod council_arbitration;
pub mod runtime_self_healing;
pub mod distributed_mercy_mesh;   // NEW in v14.0.5

pub use council_arbitration::CouncilArbitrationEngine;
pub use runtime_self_healing::RuntimeSelfHealingEngine;
pub use distributed_mercy_mesh::{DistributedMercyMesh, HealingRequest, HealingOffer, OrganismNode};

// Re-export key types for convenience
pub use runtime_self_healing::{HealthReport, Anomaly, Diagnosis, HealingAction, HealingExperience};
