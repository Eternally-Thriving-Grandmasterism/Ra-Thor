//! RiemannianMercyManifold
//!
//! Curvature-aware geometric transport for the ONE Organism.

use crate::polyhedral_harmonic_engine::U57LayerDetails;
use crate::types::EpigeneticBlessing;

#[derive(Debug, Clone)]
pub struct GeometricTransportResult {
    pub transport_applied: bool,
    pub effective_curvature: f64,
    pub coherence_after_transport: f64,
    pub suggested_blessings: Vec<EpigeneticBlessing>,
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct CurvatureParameters {
    pub base_curvature: f64,
    pub mercy_influence: f64,
    pub max_allowed_curvature: f64,
}

impl Default for CurvatureParameters {
    fn default() -> Self {
        Self {
            base_curvature: 0.82,
            mercy_influence: 0.95,
            max_allowed_curvature: 1.15,
        }
    }
}

pub struct RiemannianMercyManifold {
    pub version: &'static str,
    pub curvature_params: CurvatureParameters,
}

impl Default for RiemannianMercyManifold {
    fn default() -> Self { Self::new() }
}

impl RiemannianMercyManifold {
    pub fn new() -> Self {
        Self {
            version: "v14.4-geometric-intelligence",
            curvature_params: CurvatureParameters::default(),
        }
    }

    pub fn apply_mercy_gated_transport(
        &self,
        u57_details: &U57LayerDetails,
        base_coherence: f64,
    ) -> GeometricTransportResult {
        if !u57_details.activated {
            return GeometricTransportResult {
                transport_applied: false,
                effective_curvature: 0.0,
                coherence_after_transport: base_coherence,
                suggested_blessings: vec![],
                notes: "U57 not active".to_string(),
            };
        }

        let effective_curvature = u57_details.recommended_manifold_curvature
            * self.curvature_params.mercy_influence;

        let coherence_after = (base_coherence * 1.12).clamp(0.88, 1.35);

        GeometricTransportResult {
            transport_applied: true,
            effective_curvature,
            coherence_after_transport: coherence_after,
            suggested_blessings: vec![EpigeneticBlessing {
                blessing_type: "Riemannian_Mercy_Transport".to_string(),
                strength: coherence_after,
                target_system: "geometric".to_string(),
            }],
            notes: format!("Effective curvature: {:.3}", effective_curvature),
        }
    }
}
