//! geometric-intelligence
//!
//! Sacred Geometry + Riemannian Mercy-Gated Scoring Layer
//! for the Ra-Thor ONE Organism.
//!
//! This crate provides reusable components for:
//! - Polyhedral harmonic resonance
//! - Riemannian curvature-aware transport
//! - High-level geometric harmony scoring
//!
//! AG-SML v1.0

pub mod types;
pub mod polyhedral_harmonic_engine;
pub mod riemannian_mercy_manifold;

// Re-exports
pub use types::*;
pub use polyhedral_harmonic_engine::{PolyhedralHarmonicEngine, PolyhedralResonanceReport, U57LayerDetails};
pub use riemannian_mercy_manifold::{RiemannianMercyManifold, GeometricTransportResult};

use crate::polyhedral_harmonic_engine::PolyhedralHarmonicEngine;
use crate::riemannian_mercy_manifold::RiemannianMercyManifold;

/// High-level helper: Compute geometric harmony for a given TOLC order and coherence.
/// This is the main entry point recommended for Lattice Conductor and Real Estate use.
pub fn compute_geometric_harmony(tolc_order: u32, base_coherence: f64) -> types::GeometricHarmonyScore {
    let engine = PolyhedralHarmonicEngine::new();
    let report = engine.process_resonance(tolc_order, base_coherence);

    let manifold = RiemannianMercyManifold::new();
    let transport = if let Some(u57) = &report.u57_details {
        manifold.apply_mercy_gated_transport(u57, base_coherence)
    } else {
        types::GeometricTransportResult {
            transport_applied: false,
            effective_curvature: 0.0,
            coherence_after_transport: base_coherence,
            suggested_blessings: vec![],
            notes: "U57 not active".to_string(),
        }
    };

    types::GeometricHarmonyScore {
        multiplier: report.resonance_multiplier,
        resonance_notes: report.notes,
        active_layers: report.active_solids,
        u57_active: report.u57_potential,
    }
}
