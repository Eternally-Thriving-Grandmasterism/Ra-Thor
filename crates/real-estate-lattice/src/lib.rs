//! # Ra-Thor Real Estate Lattice (RREL)
//! Mercy-gated AGI layer for global real estate operating system.
//! PMS Bridge + Quantum Swarm Consensus + TREB/CREA MLS + Powrush-MMO hooks.

pub mod pms_bridge;
pub mod mls_integration;

pub use pms_bridge::{PmsBridge, PmsProvider, RrelError};
pub use mls_integration::{TrebMlsAdapter, MlsListing, MlsError};

pub const RREL_VERSION: &str = "0.1.0";
