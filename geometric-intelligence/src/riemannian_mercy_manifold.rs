
#[derive(Debug, Clone)]
pub struct WannierSpreadResult {
    pub total_spread: f64,
    pub invariant_spread: f64,      // Ω_I (related to Berry curvature)
    pub estimated_gauge_dependent: f64,
    pub notes: String,
}

impl RiemannianMercyManifold {

    /// Computes MLWF-like spread.
    /// invariant_spread ≈ integral of Berry curvature (Ω_I)
    pub fn compute_wannier_spread(
        &self,
        curvatures: &[f64],
        areas: &[f64],
    ) -> WannierSpreadResult {
        let invariant = self.accumulate_holonomy(curvatures, areas).abs();

        // Simple model: total spread = invariant + small gauge-dependent term
        let gauge_dependent = (curvatures.len() as f64) * 0.05;
        let total = invariant + gauge_dependent;

        let notes = if invariant > 0.8 {
            "High invariant spread. Difficult to localize (topological character).".to_string()
        } else {
            "Reasonable spread. Good localization possible.".to_string()
        };

        WannierSpreadResult {
            total_spread: total,
            invariant_spread: invariant,
            estimated_gauge_dependent: gauge_dependent,
            notes,
        }
    }
}
