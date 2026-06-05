//! powrush/src/lib.rs
//! Powrush Crate — Main library entry point

pub mod common;
pub mod server;

// Re-exports for convenience
pub use common::RbeState;
pub use server::Event;