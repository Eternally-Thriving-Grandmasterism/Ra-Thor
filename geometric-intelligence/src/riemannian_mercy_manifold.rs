
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_berry_curvature_moderate() {
        let manifold = RiemannianMercyManifold::new();
        let result = manifold.compute_berry_curvature(0.85);
        assert!(result.effective_curvature > 0.5);
        assert!(result.berry_curvature_density > 0.0);
    }

    #[test]
    fn test_compute_berry_curvature_high() {
        let manifold = RiemannianMercyManifold::new();
        let result = manifold.compute_berry_curvature(1.5);
        assert!(result.effective_curvature >= 1.0);
    }

    #[test]
    fn test_compute_berry_phase_analog_weak() {
        let manifold = RiemannianMercyManifold::new();
        let curvatures = vec![0.1, 0.2, 0.15];
        let areas = vec![0.5, 0.5, 0.5];
        let result = manifold.compute_berry_phase_analog(&curvatures, &areas);
        assert!(result.magnitude < 0.5);
    }

    #[test]
    fn test_accumulate_holonomy() {
        let manifold = RiemannianMercyManifold::new();
        let curvatures = vec![0.8, 0.9, 0.7];
        let areas = vec![1.0, 1.0, 1.0];
        let total = manifold.accumulate_holonomy(&curvatures, &areas);
        assert!(total.abs() > 0.0);
    }

    #[test]
    fn test_transport_sequence_has_accumulated_holonomy() {
        let manifold = RiemannianMercyManifold::new();
        // Create a mock U57 details
        let u57 = U57LayerDetails {
            activated: true,
            resonance_multiplier_contribution: 1.35,
            suggested_riemannian_transport_potential: 0.8,
            recommended_manifold_curvature: 0.9,
            geometric_meaning: "Test".to_string(),
            integration_notes: "Test".to_string(),
        };

        let result = manifold.run_u57_informed_transport_sequence(&u57, 0.95, 6);
        assert!(result.transport_applied);
        // Accumulated holonomy should be non-zero after several steps
        assert!(result.accumulated_holonomy.abs() > 0.01);
    }
}
