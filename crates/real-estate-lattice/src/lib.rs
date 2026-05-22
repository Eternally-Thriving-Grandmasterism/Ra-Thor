//! Real Estate Lattice (RREL) v3.1 — Eternal Organism
//! Mercy-gated, TOLC 8 enforced, PATSAGi-coordinated, RBE-aligned.
//! Part of Ra-Thor monorepo

pub mod reco_form_handlers;
pub mod rrel_brokerage_assembler;
pub mod powrush_rrel_bridge;
pub mod rrel_leptos_dashboard;

// Re-exports for clean usage across the monorepo and Tauri
pub use reco_form_handlers::*;
pub use rrel_brokerage_assembler::*;
pub use powrush_rrel_bridge::*;
pub use rrel_leptos_dashboard::*;