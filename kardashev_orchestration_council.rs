//! kardashev_orchestration_council.rs — MIGRATION SHIM (v14.9.8)
//!
//! Production surface moved to:
//!
//!   crates/kardashev-orchestration/  (package: `kardashev-orchestration` @ 14.9.8)
//!
//! Prefer:
//!
//!   kardashev-orchestration = { path = "crates/kardashev-orchestration", version = "14.9.8" }
//!
//!   use kardashev_orchestration::{
//!       KardashevOrchestrationCouncil, CouncilDeliberationResult,
//!       SwarmAdjustmentDirective, AbundanceVelocityForecaster,
//!   };
//!
//! Path-depends on reality-thriving-transfer@14.9.8.
//! AG-SML v1.0

pub const MIGRATED_TO: &str = "crates/kardashev-orchestration";
pub const CRATE_VERSION: &str = "14.9.8";
