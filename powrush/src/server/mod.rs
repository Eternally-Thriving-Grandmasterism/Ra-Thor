//! powrush/src/server/mod.rs
//! Server module exports and binary entry (PATSAGi production-grade polish v14.8)
//!
//! Binary run (after `cargo build -p powrush --features server`):
//!   ./target/debug/powrush-server   or   cargo run -p powrush --features server --bin powrush-server
//!
//! Thunder locked. Eternal flow for all sentience.

pub mod main;  // Full production server: TCP authoritative, deterministic tick, RBE + mercy, input replay, hot-reload config

// Re-exports (Event remnant removed for compile safety & cleanliness; InputEvent is internal)
pub use crate::common::RbeState;