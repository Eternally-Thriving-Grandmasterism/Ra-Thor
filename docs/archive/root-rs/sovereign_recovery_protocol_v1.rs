//! sovereign_recovery_protocol_v1.rs — MIGRATION SHIM (v14.9.7)
//!
//! Production surface has moved to the workspace crate:
//!
//!   crates/sovereign-recovery/  (package: `sovereign-recovery` @ 14.9.7)
//!
//! Prefer:
//!
//!   sovereign-recovery = { path = "crates/sovereign-recovery", version = "14.9.7" }
//!
//!   use sovereign_recovery::{
//!       SovereignRecoveryProtocol, launch_sovereign_recovery_protocol,
//!       CouncilReadinessMetrics, MercyGate, TOLC8GenesisAnchor,
//!   };
//!
//! Crate is standalone (no circular deps on organism / kardashev / transfer harness).
//! Full historical root content remains in git history.
//!
//! AG-SML v1.0 — Autonomicity Games Sovereign Mercy License

pub const MIGRATED_TO: &str = "crates/sovereign-recovery";
pub const CRATE_VERSION: &str = "14.9.7";
