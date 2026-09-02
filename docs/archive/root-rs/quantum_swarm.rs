//! quantum_swarm.rs — MIGRATION SHIM (v14.9.5)
//!
//! Production surface has moved to the workspace crate:
//!
//!   crates/quantum-swarm/  (package: `quantum-swarm` @ 14.9.5)
//!
//! Prefer:
//!
//!   quantum-swarm = { path = "crates/quantum-swarm", version = "14.9.5" }
//!   // optional GPU benchmarks:
//!   quantum-swarm = { path = "crates/quantum-swarm", features = ["gpu"] }
//!
//!   use quantum_swarm::{
//!       QuantumSwarmEngine, QuantumSwarmConfig, QuantumSwarmMember,
//!       CouncilReadinessMetrics,
//!   };
//!
//! Crate includes lightweight in-crate SovereignRecoveryProtocol so it
//! compiles standalone. Full root sovereign_recovery_protocol_v1 +
//! reality_thriving_transfer_harness remain available for deep ONE Organism
//! wiring.
//!
//! AG-SML v1.0 — Autonomicity Games Sovereign Mercy License

pub const MIGRATED_TO: &str = "crates/quantum-swarm";
pub const CRATE_VERSION: &str = "14.9.5";
