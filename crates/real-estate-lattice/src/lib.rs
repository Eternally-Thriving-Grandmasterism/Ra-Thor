//! # Ra-Thor Real Estate Lattice (RREL)
//! Mercy-gated AGI layer for global real estate operating system.
//! PMS Bridge + Quantum Swarm Consensus + Powrush-MMO hooks.

pub mod pms_bridge;

pub use pms_bridge::{PmsBridge, PmsProvider, RrelError};

pub const RREL_VERSION: &str = "0.1.0";
