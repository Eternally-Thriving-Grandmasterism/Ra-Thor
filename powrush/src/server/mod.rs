//! powrush/src/server/mod.rs
//! Server-specific modules and re-exports

pub mod main;  // The main server binary logic

// Re-export key types for convenience
pub use crate::common::RbeState;
pub use crate::Event;