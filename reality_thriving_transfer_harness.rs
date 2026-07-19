//! reality_thriving_transfer_harness.rs — MIGRATION SHIM (v14.9.8)
//!
//! Production surface moved to:
//!
//!   crates/reality-thriving-transfer/  (package: `reality-thriving-transfer` @ 14.9.8)
//!
//! Prefer:
//!
//!   reality-thriving-transfer = { path = "crates/reality-thriving-transfer", version = "14.9.8" }
//!
//!   use reality_thriving_transfer::{
//!       RealityThrivingTransferCalculator, RealityThrivingTransferScore,
//!       PowrushTelemetry, run_quantum_swarm_v2_kardashev_benchmark,
//!   };
//!
//! AG-SML v1.0

pub const MIGRATED_TO: &str = "crates/reality-thriving-transfer";
pub const CRATE_VERSION: &str = "14.9.8";
