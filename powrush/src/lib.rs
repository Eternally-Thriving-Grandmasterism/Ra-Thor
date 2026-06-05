//! powrush/src/lib.rs
//! Powrush Crate — Main library entry point (PATSAGi v14.8 production polish)
//!
//! Features:
//!   - server: enables powrush-server binary (TCP MMO for humans to play now)
//!   - client: future WebSocket/browser client
//!   - full: both
//!
//! AG-SML v1.0 | Mercy-gated | RBE abundance for all factions

pub mod common;
pub mod server;

// Re-exports (Event removed - was undefined remnant; only valid shared types)
pub use common::RbeState;