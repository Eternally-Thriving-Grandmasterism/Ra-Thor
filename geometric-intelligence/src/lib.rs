//! geometric-intelligence
//!
//! Sacred Geometry + Riemannian Mercy-Gated Scoring Layer
//! for the Ra-Thor ONE Organism.

pub mod types;
pub mod polyhedral_harmonic_engine;
pub mod riemannian_mercy_manifold;

pub use types::*;
pub use polyhedral_harmonic_engine::{PolyhedralHarmonicEngine, PolyhedralResonanceReport, U57LayerDetails};
pub use riemannian_mercy_manifold::{RiemannianMercyManifold, GeometricTransportResult};

use crate::polyhedral_harmonic_engine::PolyhedralHarmonicEngine;
use crate::riemannian_mercy_manifold::RiemannianMercyManifold;

/// High-level helper: Compute geometric harmony for a given TOLC order and coherence.
pub fn compute_geometric_harmony(tolc_order: u32, base_coherence: f64) -> types::GeometricHarmonyScore {
    let engine = PolyhedralHarmonicEngine::new();
    let report = engine.process_resonance(tolc_order, base_coherence);

    let manifold = RiemannianMercyManifold::new();
    let _transport = if let Some(u57) = &report.u57_details {
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

/// Free function wrapper for applying U57 Riemannian transport.
/// Creates a default manifold internally for maximum ergonomics.
pub fn apply_u57_riemannian_transport(
    polyhedral_report: &PolyhedralResonanceReport,
    base_coherence: f64,
) -> Option<GeometricTransportResult> {
    let manifold = RiemannianMercyManifold::new();
    manifold.apply_u57_riemannian_transport(polyhedral_report, base_coherence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyhedral_basic() {
        let engine = PolyhedralHarmonicEngine::new();
        let report = engine.process_resonance(8, 0.95);
        assert!(report.resonance_multiplier >= 1.0);
        assert!(!report.u57_potential);
    }

    #[test]
    fn test_polyhedral_u57_activation() {
        let engine = PolyhedralHarmonicEngine::new();
        let report = engine.process_resonance(150, 0.95);
        assert!(report.u57_potential);
        assert!(report.u57_details.is_some());
    }

    #[test]
    fn test_compute_geometric_harmony() {
        let score = compute_geometric_harmony(55, 0.92);
        assert!(score.multiplier > 1.0);
        assert!(!score.u57_active);
    }

    #[test]
    fn test_geometric_harmony_high_tolc() {
        let score = compute_geometric_harmony(200, 0.90);
        assert!(score.u57_active);
    }

    #[test]
    fn test_apply_u57_riemannian_transport_inactive() {
        let engine = PolyhedralHarmonicEngine::new();
        let report = engine.process_resonance(55, 0.95); // U57 not active
        let result = apply_u57_riemannian_transport(&report, 0.95);
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_u57_riemannian_transport_active() {
        let engine = PolyhedralHarmonicEngine::new();
        let report = engine.process_resonance(180, 0.92); // U57 should be active
        let result = apply_u57_riemannian_transport(&report, 0.92);
        assert!(result.is_some());
        if let Some(transport) = result {
            assert!(transport.transport_applied);
            assert!(transport.coherence_after_transport > 0.88);
        }
    }
}
