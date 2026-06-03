// crates/real-estate-lattice/src/geometric_harmony_advisor.rs
//
// GeometricHarmonyAdvisor v14.4 — Sacred Geometry for Sovereign Real Estate
// ONE Organism + Real Estate Lattice Integration
//
// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
// Brings PolyhedralHarmonicEngine + U57 readiness into USA Pilot and global real estate systems.
//
// Philosophy:
// Property is not just economic — it is geometric and energetic.
// This advisor uses polyhedral resonance and (when active) U57 Riemannian principles
// to evaluate spatial harmony, suggest optimal layouts, and support mercy-aligned
// valuation and stewardship decisions.

use quantum_swarm_orchestrator::PolyhedralHarmonicEngine;
use quantum_swarm_orchestrator::U57LayerDetails;

/// Result of a geometric harmony assessment on a property or site.
#[derive(Debug, Clone)]
pub struct GeometricHarmonyAssessment {
    pub resonance_multiplier: f64,
    pub dominant_solids: Vec<String>,
    pub u57_active: bool,
    pub harmony_score: f64,           // 0.0 – 1.0+ (higher = more harmonious)
    pub recommended_layout_notes: String,
    pub u57_transport_potential: Option<f64>,
}

/// Advisor that brings ONE Organism geometric intelligence into Real Estate Lattice.
pub struct GeometricHarmonyAdvisor {
    pub version: &'static str,
    polyhedral_engine: PolyhedralHarmonicEngine,
}

impl GeometricHarmonyAdvisor {
    pub fn new() -> Self {
        Self {
            version: "v14.4-initial",
            polyhedral_engine: PolyhedralHarmonicEngine::new(),
        }
    }

    /// Assess geometric harmony of a property/site using current TOLC order.
    pub fn assess_property_harmony(
        &self,
        tolc_order: u32,
        base_coherence: f64,
    ) -> GeometricHarmonyAssessment {
        let report = self.polyhedral_engine.process_resonance(tolc_order, base_coherence);

        let harmony_score = (report.resonance_multiplier * 0.65).clamp(0.75, 1.45);

        let recommended_notes = if report.u57_potential {
            "U57 potential active. Consider non-Euclidean / curvature-aware spatial design principles for highest harmony.".to_string()
        } else if tolc_order >= 34 {
            "Strong Catalan dual resonance detected. Recommend balanced, reciprocal spatial layouts.".to_string()
        } else {
            "Classical Platonic harmony dominant. Prioritize symmetry, proportion, and elemental balance.".to_string()
        };

        GeometricHarmonyAssessment {
            resonance_multiplier: report.resonance_multiplier,
            dominant_solids: report.active_solids,
            u57_active: report.u57_potential,
            harmony_score,
            recommended_layout_notes: recommended_notes,
            u57_transport_potential: report.u57_details.as_ref().map(|d| d.suggested_riemannian_transport_potential),
        }
    }
}