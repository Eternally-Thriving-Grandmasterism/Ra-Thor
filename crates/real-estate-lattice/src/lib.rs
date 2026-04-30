//! # Ra-Thor Real Estate Lattice (RREL)
//! Mercy-gated AGI layer for global real estate operating system.
//! PMS Bridge + Quantum Swarm Consensus + TREB/CREA MLS + RECO/LAT/Divisional Court Enforcement

pub mod pms_bridge;
pub mod mls_integration;
pub mod reco_enforcement;

pub use pms_bridge::{PmsBridge, PmsProvider, RrelError};
pub use mls_integration::{TrebMlsAdapter, MlsListing, MlsError};
pub use reco_enforcement::{RecoEnforcementEngine, RecoEnforcementAction, RecoError};

pub const RREL_VERSION: &str = "0.1.0";
