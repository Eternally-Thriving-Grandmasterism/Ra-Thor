
    #[test]
    fn test_wannier_spread() {
        let manifold = RiemannianMercyManifold::new();
        let curvatures = vec![0.7, 0.8, 0.9];
        let areas = vec![1.0, 1.0, 1.0];

        let spread = manifold.compute_wannier_spread(&curvatures, &areas);

        assert!(spread.invariant_spread >= 0.0);
        assert!(spread.total_spread >= spread.invariant_spread);
    }

    #[test]
    fn test_wannier_spread_high_curvature() {
        let manifold = RiemannianMercyManifold::new();
        let curvatures = vec![1.2, 1.3, 1.4];
        let areas = vec![1.0, 1.0, 1.0];

        let spread = manifold.compute_wannier_spread(&curvatures, &areas);
        // High curvature should lead to higher invariant spread
        assert!(spread.invariant_spread > 0.5);
    }
