/// RREL v3.2 Eternal Organism
/// Real Estate Lattice — Mercy-gated, TOLC 8 enforced, PATSAGi-aligned, Lattice Conductor + Quantum Swarm ready.
///
/// Archived note (for future reference):
/// Previous v2.1 contained experimental hyperbolic/Möbius spatial valuation using complex numbers and gyrovector distance.
/// That logic was exploratory and has been superseded by properly mercy-gated, council-voted valuation systems.
/// Preserved conceptually for potential advanced spatial math modules later if aligned with TOLC 8 and RBE principles.
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