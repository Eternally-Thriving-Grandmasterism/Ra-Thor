    #[test]
    fn tikhonov_lambda_is_zero_on_canonical() {
        let b = LivingMercyBasis::canonical();
        assert_eq!(b.tikhonov_lambda, 0.0);
        assert!(b.gram_residual() < 1e-15);
        assert_eq!(schedule_tikhonov_lambda(0.0, 0.0), 0.0);
        let p = b.projector_matrix();
        let p_exact = b.projector_matrix_exact();
        assert!((p - p_exact).norm() < 1e-15);
    }

    #[test]
    fn tikhonov_damped_projector_is_symmetric_psd() {
        let mut b = LivingMercyBasis::canonical();
        b.e[(0, 1)] += 1e-3;
        b.e[(4, 7)] -= 2e-3;
        b.reschedule_tikhonov(0.0);
        assert!(b.tikhonov_lambda > 0.0);
        let proj = MercyProjector { basis: b };
        assert!(proj.verify_symmetry(1e-10));
        assert!(proj.verify_symmetric_psd(1e-9));
        let p = proj.basis.projector_matrix();
        assert!((&p * &p - &p).norm().is_finite());
    }

    #[test]
    fn tikhonov_lambda_rises_with_drift_and_resets_on_purify() {
        let mut zone = ZoneState::new(0);
        let before = zone.basis.tikhonov_lambda;
        zone.inject_drift(1e-2);
        let drifted = zone.basis.tikhonov_lambda;
        assert!(drifted > before);
        assert!(drifted > 1e-4);
        let rho = zone.purify();
        assert!(rho < 1e-10);
        assert!(zone.basis.tikhonov_lambda < 1e-9);
        assert!(zone.basis.gram_residual() < 1e-10);
    }

    #[test]
    fn tikhonov_stabilizes_near_singular_gram() {
        let mut b = LivingMercyBasis::canonical();
        for i in 0..AMBIENT_DIM {
            b.e[(i, 1)] = b.e[(i, 0)];
        }
        assert!(b.projector_matrix_exact().iter().all(|x| x.is_finite()));
        b.tikhonov_lambda = 1e-4;
        let p = b.projector_matrix_tikhonov(b.tikhonov_lambda);
        assert!(p.iter().all(|x| x.is_finite()));
        assert!((&p - p.transpose()).norm() < 1e-10);
        let eig = nalgebra::SymmetricEigen::new((p + p.transpose()) * 0.5);
        assert!(eig.eigenvalues.iter().all(|ev| *ev >= -1e-8));
    }

    #[test]
    fn tikhonov_stress_gain_schedules_lambda() {
        let lam_rho = schedule_tikhonov_lambda(0.01, 0.0);
        let lam_stress = schedule_tikhonov_lambda(0.0, 1_000.0);
        let lam_both = schedule_tikhonov_lambda(0.01, 1_000.0);
        assert!((lam_rho - 0.01).abs() < 1e-15);
        assert!((lam_stress - TIKHONOV_STRESS_GAIN * 1_000.0).abs() < 1e-18);
        assert!((lam_both - (lam_rho + lam_stress)).abs() < 1e-15);
    }

    #[test]
    fn concurrent_zones_global_purify_clears_rho() {
        let mut lattice = ConcurrentZoneLattice::new(4);
        for rho in lattice.global_purify() { assert!(rho < 1e-10); }
        assert!(lattice.max_rho() < 1e-10);
    }
