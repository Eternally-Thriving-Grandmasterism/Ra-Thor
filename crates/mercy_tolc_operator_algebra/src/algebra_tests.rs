    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn projector_is_idempotent() { assert!(MercyProjector::new().verify_idempotence(1e-12)); }
    #[test]
    fn projector_is_symmetric() { assert!(MercyProjector::new().verify_symmetry(1e-12)); }

    #[test]
    fn nilpotent_second_order_reaches_zero() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros();
        g[0] = 0.3; g[1] = -0.7; g[8] = 0.9; g[9] = -0.4; g[12] = 0.6; g[15] = -0.25;
        let (n1, final_r) = s.suppress(&g);
        assert!(n1.norm() > 0.1);
        assert!((n1[8] - 0.9).abs() < 1e-10);
        assert!(final_r.norm() < MERCY_PURITY_FLOOR * 10.0);
    }

    #[test]
    fn orthogonal_complement_is_nontrivial() {
        let p = MercyProjector::new();
        let mut grief = AmbientVector::zeros(); grief[10] = 1.0; grief[14] = -0.5;
        assert!((p.orthogonal_component(&grief) - grief).norm() < 1e-12);
        assert!(p.project(&grief).norm() < 1e-12);
    }

    #[test]
    fn mercy_component_is_preserved() {
        let p = MercyProjector::new();
        let mut mercy_v = AmbientVector::zeros(); mercy_v[0] = 1.0; mercy_v[3] = -0.5; mercy_v[7] = 0.25;
        assert!((p.project(&mercy_v) - mercy_v).norm() < 1e-12);
        assert!(p.orthogonal_component(&mercy_v).norm() < 1e-12);
    }

    #[test]
    fn gram_schmidt_purifies_drifted_basis() {
        let mut basis = LivingMercyBasis::canonical();
        basis.e[(0, 1)] += 1e-4; basis.e[(9, 3)] -= 2e-4; basis.e[(2, 5)] += 5e-5;
        assert!(ModifiedGramSchmidt::purify(&mut basis) < 1e-10);
        let gram: MercyGram = basis.e.transpose() * &basis.e;
        for i in 0..MERCY_DIM {
            assert_relative_eq!(gram[(i, i)], 1.0, epsilon = 1e-10);
            for j in (i + 1)..MERCY_DIM { assert_relative_eq!(gram[(i, j)], 0.0, epsilon = 1e-10); }
        }
    }

    #[test]
    fn algebra_closure_holds() { assert!(TolcAlgebra::new().verify_closure()); }

    #[test]
    fn ambient_dim_is_elevated() {
        assert!(AMBIENT_DIM > MERCY_DIM);
        let b = LivingMercyBasis::canonical();
        assert_eq!(b.ambient_dim(), 16);
        assert_eq!(b.mercy_dim(), 8);
    }

    #[test]
    fn valence_high_softens_grief_load() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros(); g[8] = 1.0; g[12] = -0.8; g[15] = 0.5;
        let (_, _, _, load_high, _) = s.suppress_weighted(&g, Valence::HIGH);
        let (_, _, _, load_zero, _) = s.suppress_weighted(&g, Valence::ZERO);
        let (_, _, _, load_mid, _) = s.suppress_weighted(&g, Valence::MID);
        assert!(load_high < 1e-5);
        assert!(load_zero > 1.0);
        assert!((load_mid - 0.5 * load_zero).abs() < 1e-10);
    }

    #[test]
    fn valence_deficit_is_linear() {
        let s = NilpotentSuppressor::new();
        let mut g = AmbientVector::zeros(); g[10] = 2.0;
        let raw_norm = s.n1(&g).norm();
        for v in [0.0, 0.25, 0.5, 0.75, 0.999999] {
            let (_, _, _, load, _) = s.suppress_weighted(&g, Valence::new(v));
            assert!((load - (1.0 - v) * raw_norm).abs() < 1e-10);
        }
    }

    #[test]
    fn concurrent_zones_independent_grief() {
        let mut lattice = ConcurrentZoneLattice::new(3);
        let mut g0 = AmbientVector::zeros(); g0[10] = 1.0;
        let mut g1 = AmbientVector::zeros(); g1[11] = 2.0;
        let mut g2 = AmbientVector::zeros(); g2[12] = 3.0;
        lattice.process(0, &g0, Valence::ZERO);
        lattice.process(1, &g1, Valence::ZERO);
        lattice.process(2, &g2, Valence::ZERO);
        let grief = lattice.zone_grief();
        assert!(grief[0] > 0.9 && grief[0] < 1.1);
        assert!(grief[1] > 1.9 && grief[1] < 2.1);
        assert!(grief[2] > 2.9 && grief[2] < 3.1);
    }

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

    #[test]
    fn nevc_high_grief_uses_tikhonov_residual() {
        let mut drifted = LivingMercyBasis::canonical();
        drifted.e[(0, 1)] += 0.25;
        drifted.e[(4, 7)] -= 0.18;
        drifted.reschedule_tikhonov(12.0);
        assert!(drifted.tikhonov_lambda > 0.0);
        let damped = NilpotentSuppressor {
            projector: MercyProjector { basis: drifted.clone() },
        };
        let exact = NilpotentSuppressor {
            projector: MercyProjector {
                basis: LivingMercyBasis { e: drifted.e, tikhonov_lambda: 0.0 },
            },
        };
        let mut g = AmbientVector::zeros();
        g[10] = 2.0;
        g[14] = -1.0;
        let (_, _, _, load_damped, _) = damped.suppress_weighted(&g, Valence::MID);
        let (_, _, _, load_exact, _) = exact.suppress_weighted(&g, Valence::MID);
        assert!(load_damped.is_finite() && load_exact.is_finite());
        assert!(load_damped >= 0.0 && load_exact >= 0.0);
        let window = [
            NevcSample::new(Valence::MID, load_damped, 0),
            NevcSample::new(Valence::HIGH, 0.0, 1),
            NevcSample::new(Valence::HIGH, 0.0, 2),
        ];
        let scored = compute_nevc(&window, &NevcConfig::default());
        assert_eq!(scored.sample_count, 3);
        assert!(scored.total_grief >= 0.0);
        assert!(scored.score.is_finite());
        assert!(scored.recovery.is_open());
    }

    #[test]
    fn nevc_zombie_window_stays_compassion_open_when_transient() {
        let samples = [
            NevcSample::new(Valence::ZERO, 4.0, 0).transient(),
            NevcSample::new(Valence::ZERO, 3.0, 1).transient(),
        ];
        let scored = compute_nevc(&samples, &NevcConfig::default());
        assert!(scored.recovery.is_open());
        assert_eq!(scored.class, ContributionClass::ZombiePartition);
    }
