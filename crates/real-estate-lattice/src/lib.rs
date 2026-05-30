//! RREL v3.2 Eternal Organism
//! Real Estate Lattice — Mercy-gated, TOLC 8 enforced, PATSAGi-aligned, Lattice Conductor + Quantum Swarm ready.

pub mod offer_package;
pub mod compliance_helpers;
pub mod rrel_brokerage_assembler;
pub mod powrush_rrel_bridge;
pub mod reco_form_handlers;
pub mod rrel_leptos_dashboard;
pub mod rrel_lattice_conductor_bridge;
pub mod rrel_quantum_swarm_participant;

// v14.3 Execution Stabilization additions
pub mod property_type_classifier;
pub mod deal_type_classifier;
pub mod form_mapping_engine;
pub mod offer_package_validator;
pub mod multi_offer_track_engine;
pub mod disclosure_manager;
pub mod lawyer_due_diligence_generator;
pub mod lawyer_report_pdf_generator;

// New micro-expansion modules
pub mod rrel_offer_risk_summary;

pub use offer_package::*;
pub use compliance_helpers::*;
pub use rrel_brokerage_assembler::*;
pub use powrush_rrel_bridge::*;
pub use reco_form_handlers::*;
pub use rrel_leptos_dashboard::*;
pub use rrel_lattice_conductor_bridge::*;
pub use rrel_quantum_swarm_participant::*;

// Re-exports for v14.3 modules
pub use property_type_classifier::*;
pub use deal_type_classifier::*;
pub use form_mapping_engine::*;
pub use offer_package_validator::*;
pub use multi_offer_track_engine::*;
pub use disclosure_manager::*;
pub use lawyer_due_diligence_generator::*;
pub use lawyer_report_pdf_generator::*;

// New helper
pub use rrel_offer_risk_summary::*;

#[cfg(feature = "leptos")]
pub use leptos;

#[cfg(feature = "tauri")]
pub use tauri;