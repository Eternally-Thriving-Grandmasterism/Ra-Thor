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
