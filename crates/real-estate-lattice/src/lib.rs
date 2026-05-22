//! RREL v3.2 Eternal Organism — Full module wiring + re-exports
//! TOLC 8, MercyAligned, PATSAGi, Lattice Conductor ready.

pub mod offer_package;
pub mod compliance_helpers;
pub mod rrel_brokerage_assembler;
pub mod powrush_rrel_bridge;
pub mod reco_form_handlers;
pub mod rrel_leptos_dashboard;
pub mod rrel_lattice_conductor_bridge;
pub mod rrel_quantum_swarm_participant;

pub use offer_package::*;
pub use compliance_helpers::*;
pub use rrel_brokerage_assembler::*;
pub use powrush_rrel_bridge::*;
pub use reco_form_handlers::*;
pub use rrel_leptos_dashboard::*;
pub use rrel_lattice_conductor_bridge::*;
pub use rrel_quantum_swarm_participant::*;

#[cfg(feature = "leptos")]
pub use leptos;

#[cfg(feature = "tauri")]
pub use tauri;