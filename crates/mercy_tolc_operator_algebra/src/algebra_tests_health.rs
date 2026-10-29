    #[test]
    fn valence_clamps_to_unit_interval() {
        assert_eq!(Valence::new(-1.0).value(), 0.0);
        assert_eq!(Valence::new(2.0).value(), 1.0);
        assert!(Valence::HIGH.is_high());
    }

    #[test]
    fn adaptive_floor_tightens_with_high_valence() {
        let pure = adaptive_purity_floor(Valence::new(1.0));
        let high = adaptive_purity_floor(Valence::HIGH);
        let zero = adaptive_purity_floor(Valence::ZERO);
        assert!((pure - MERCY_PURITY_FLOOR).abs() < 1e-18);
        assert!(high >= pure && zero > high);
        assert!((zero - MERCY_PURITY_FLOOR * 100.0).abs() < 1e-15);
    }

    #[test]
    fn suppress_weighted_reports_under_floor() {
        let s = NilpotentSuppressor::new();
        let mut tiny = AmbientVector::zeros(); tiny[10] = 1e-12;
        let (_, _, _, _, under_h) = s.suppress_weighted(&tiny, Valence::HIGH);
        assert!(under_h);
        let mut big = AmbientVector::zeros(); big[10] = 2.0;
        let (_, _, _, load_z, under_z) = s.suppress_weighted(&big, Valence::ZERO);
        assert!(!under_z && load_z > 1.0);
    }

    #[test]
    fn concurrent_zones_staggered_tick_runs() {
        let mut lattice = ConcurrentZoneLattice::new(2);
        lattice.purify_period = 10;
        let mut g = AmbientVector::zeros(); g[10] = 0.5;
        for i in 0..25 { lattice.process(i % 2, &g, Valence::MID); }
        assert!(lattice.total_grief() > 0.0);
        assert_eq!(lattice.zones[0].vectors_processed + lattice.zones[1].vectors_processed, 25);
    }

    #[test]
    fn soft_feedback_bridge_records_events() {
        let mut bridge = SoftFeedbackBridge::new(3);
        let ev = bridge.ingest_scalar_grief(1, 2.0, Valence::ZERO);
        assert_eq!(ev.zone_id, 1);
        assert!(ev.grief_load > 1.5);
        assert!(!ev.under_floor);
        let soft = bridge.ingest_scalar_grief(0, 2.0, Valence::HIGH);
        assert!(soft.grief_load < 1e-5);
        assert_eq!(bridge.drain_events().len(), 2);
        assert!(bridge.drain_events().is_empty());
        let snaps = bridge.snapshots();
        assert_eq!(snaps.len(), 3);
        assert!(snaps[1].grief_absorbed > 1.5);
    }

    #[test]
    fn soft_feedback_event_fields_are_stable() {
        let ev = SoftFeedbackEvent {
            zone_id: 2, grief_load: 1.25, valence: 0.5, under_floor: false, tick: 42,
        };
        assert_eq!(ev.zone_id, 2);
        assert!((ev.grief_load - 1.25).abs() < 1e-12);
        assert!((ev.valence - 0.5).abs() < 1e-12);
        assert!(!ev.under_floor);
        assert_eq!(ev.tick, 42);
    }

    #[test]
    fn lattice_health_report_is_healthy_after_purify() {
        let mut bridge = SoftFeedbackBridge::new(3);
        for i in 0..30 { bridge.ingest_scalar_grief(i % 3, 1.0, Valence::MID); }
        bridge.global_purify();
        let h = bridge.health_report();
        assert_eq!(h.schema, "ra_thor_lattice_health_v1");
        assert_eq!(h.zone_count, 3);
        assert!(h.total_vectors >= 30);
        assert!(h.healthy, "max_rho={}", h.max_rho);
        assert!(h.max_rho < 1e-9);
    }

    #[test]
    fn adaptive_purify_period_tightens_under_grief() {
        let mut lattice = ConcurrentZoneLattice::new(2);
        lattice.purify_period = 1000;
        lattice.adaptive_grief_scale = 100.0;
        lattice.min_purify_period = 10;
        let base = lattice.effective_purify_period(0);
        assert_eq!(base, 1000);
        let mut g = AmbientVector::zeros(); g[10] = 5.0;
        for _ in 0..50 { lattice.process(0, &g, Valence::ZERO); }
        let tight = lattice.effective_purify_period(0);
        assert!(tight < base);
        assert!(tight >= lattice.min_purify_period);
        assert!(lattice.effective_purify_period(1) >= tight);
    }

    #[test]
    fn high_grief_zone_fires_more_cosmic_ticks() {
        let mut lattice = ConcurrentZoneLattice::new(2);
        lattice.purify_period = 100;
        lattice.adaptive_grief_scale = 20.0;
        lattice.min_purify_period = 10;
        lattice.stress_alpha = 0.15;
        let mut heavy = AmbientVector::zeros(); heavy[10] = 5.0;
        let mut light = AmbientVector::zeros(); light[11] = 0.01;
        for _ in 0..800 { lattice.process(0, &heavy, Valence::ZERO); }
        for _ in 0..800 { lattice.process(1, &light, Valence::HIGH); }
        let c0 = lattice.zones[0].purify_count;
        let c1 = lattice.zones[1].purify_count;
        assert!(c0 > 0);
        assert!(c0 > c1, "z0={c0} z1={c1}");
    }

    #[test]
    fn zone_snapshot_exposes_observability_fields() {
        let mut bridge = SoftFeedbackBridge::new(2);
        bridge.ingest_scalar_grief(0, 1.0, Valence::ZERO);
        bridge.global_purify();
        let snaps = bridge.snapshots();
        assert_eq!(snaps.len(), 2);
        assert!(snaps[0].purify_count >= 1);
        assert!(snaps[0].effective_period >= 1);
        let h = bridge.health_report();
        assert!(h.zones[0].purify_count >= 1);
    }

    #[test]
    fn health_report_exposes_stress_aggregates() {
        let mut bridge = SoftFeedbackBridge::new(3);
        for i in 0..60 { bridge.ingest_scalar_grief(i % 3, 2.0, Valence::ZERO); }
        bridge.global_purify();
        let h = bridge.health_report();
        assert!(h.total_purify_count >= 3);
        assert!(h.max_stress_ema > 0.0);
        assert!(h.mean_effective_period >= 1.0);
        assert_eq!(h.zone_count, 3);
        assert!(h.total_vectors >= 60);
        assert!(h.health_score > 0.0 && h.health_score <= 1.0);
        assert!(h.healthy);
    }

    #[test]
    fn zone_health_status_classifies_calm_as_healthy() {
        assert_eq!(ZoneHealthStatus::classify(0.0, 0.0, 500.0), ZoneHealthStatus::Healthy);
        assert_eq!(ZoneHealthStatus::classify(60.0, 0.0, 500.0), ZoneHealthStatus::Stressed);
        assert_eq!(ZoneHealthStatus::classify(500.0, 0.0, 500.0), ZoneHealthStatus::Critical);
        assert_eq!(ZoneHealthStatus::classify(0.0, 1e-5, 500.0), ZoneHealthStatus::Critical);
    }

    #[test]
    fn valence_histogram_tracks_bands() {
        let mut bridge = SoftFeedbackBridge::new(1);
        bridge.ingest_scalar_grief(0, 1.0, Valence::HIGH);
        bridge.ingest_scalar_grief(0, 1.0, Valence::HIGH);
        bridge.ingest_scalar_grief(0, 1.0, Valence::MID);
        bridge.ingest_scalar_grief(0, 1.0, Valence::ZERO);
        let h = bridge.health_report();
        assert_eq!(h.valence_high_count, 2);
        assert_eq!(h.valence_mid_count, 1);
        assert_eq!(h.valence_low_count, 1);
        assert!((h.valence_mercy_ratio - 0.5).abs() < 1e-12);
    }

    #[test]
    fn valence_mercy_ratio_is_one_when_all_high() {
        let mut bridge = SoftFeedbackBridge::new(1);
        for _ in 0..5 { bridge.ingest_scalar_grief(0, 0.1, Valence::HIGH); }
        let h = bridge.health_report();
        assert_eq!(h.valence_high_count, 5);
        assert!((h.valence_mercy_ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn soft_remediate_fires_on_stressed_zone() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.adaptive_grief_scale = 10.0;
        lattice.stress_alpha = 0.25;
        lattice.soft_remediate_stressed = true;
        lattice.soft_remediate_alpha = 0.05;
        lattice.critical_auto_remediate = false;
        lattice.purify_period = 10_000;
        let mut g = AmbientVector::zeros();
        g[10] = 2.0;
        for _ in 0..30 { lattice.process(0, &g, Valence::ZERO); }
        let stress = lattice.zones[0].stress_ema;
        assert!(lattice.zones[0].soft_remediate_count > 0, "soft_remediate_count=0 stress_ema={stress}");
        assert_eq!(lattice.zones[0].critical_auto_purify_count, 0);
        assert!(lattice.total_soft_remediates() > 0);
        assert!(stress < lattice.adaptive_grief_scale, "stress escalated to Critical: {stress}");
    }

    #[test]
    fn soft_remediate_can_be_disabled() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.adaptive_grief_scale = 50.0;
        lattice.stress_alpha = 0.3;
        lattice.soft_remediate_stressed = false;
        lattice.critical_auto_remediate = false;
        lattice.purify_period = 10_000;
        let mut g = AmbientVector::zeros();
        g[10] = 3.0;
        for _ in 0..20 { lattice.process(0, &g, Valence::ZERO); }
        assert_eq!(lattice.zones[0].soft_remediate_count, 0);
    }

    #[test]
    fn critical_auto_remediate_fires_under_extreme_stress() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.adaptive_grief_scale = 5.0;
        lattice.stress_alpha = 0.5;
        lattice.critical_auto_remediate = true;
        lattice.purify_period = 10_000;
        let mut heavy = AmbientVector::zeros();
        heavy[10] = 20.0;
        for _ in 0..40 { lattice.process(0, &heavy, Valence::ZERO); }
        assert!(lattice.zones[0].critical_auto_purify_count > 0);
        assert!(lattice.total_critical_auto_purifies() > 0);
        assert!(lattice.zones[0].last_rho < 1e-9);
    }

    #[test]
    fn critical_auto_remediate_can_be_disabled() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.adaptive_grief_scale = 5.0;
        lattice.stress_alpha = 0.5;
        lattice.critical_auto_remediate = false;
        lattice.purify_period = 10_000;
        let mut heavy = AmbientVector::zeros();
        heavy[10] = 20.0;
        for _ in 0..40 { lattice.process(0, &heavy, Valence::ZERO); }
        assert_eq!(lattice.zones[0].critical_auto_purify_count, 0);
    }

    #[test]
    fn health_report_counts_zone_status() {
        let mut bridge = SoftFeedbackBridge::new(2);
        bridge.lattice.adaptive_grief_scale = 10.0;
        bridge.lattice.stress_alpha = 0.3;
        for _ in 0..30 { bridge.ingest_scalar_grief(0, 8.0, Valence::ZERO); }
        for _ in 0..5 { bridge.ingest_scalar_grief(1, 1e-12, Valence::HIGH); }
        bridge.global_purify();
        let h = bridge.health_report();
        assert_eq!(h.zones.len(), 2);
        assert!(h.zones_stressed + h.zones_critical + h.zones_healthy == 2);
        assert!(h.zones[0].status != ZoneHealthStatus::Healthy);
        assert_eq!(h.zones[1].status, ZoneHealthStatus::Healthy);
    }

    #[test]
    fn health_score_drops_under_stress_and_recovers() {
        let mut bridge = SoftFeedbackBridge::new(1);
        bridge.lattice.adaptive_grief_scale = 10.0;
        bridge.lattice.stress_alpha = 0.25;
        bridge.lattice.purify_period = 1000;
        let calm = bridge.health_report().health_score;
        assert!((calm - 1.0).abs() < 1e-6);
        for _ in 0..40 { bridge.ingest_scalar_grief(0, 8.0, Valence::ZERO); }
        bridge.global_purify();
        let stressed = bridge.health_report().health_score;
        assert!(stressed < calm && stressed > 0.0);
        for _ in 0..200 { bridge.ingest_scalar_grief(0, 1e-12, Valence::HIGH); }
        bridge.global_purify();
        let recovered = bridge.health_report().health_score;
        assert!(recovered > stressed);
    }

    #[test]
    fn stress_ema_recovers_period_under_calm() {
        let mut lattice = ConcurrentZoneLattice::new(1);
        lattice.purify_period = 1000;
        lattice.adaptive_grief_scale = 10.0;
        lattice.min_purify_period = 20;
        lattice.stress_alpha = 0.2;
        let mut heavy = AmbientVector::zeros(); heavy[10] = 8.0;
        for _ in 0..40 { lattice.process(0, &heavy, Valence::ZERO); }
        let tight = lattice.effective_purify_period(0);
        assert!(tight < 1000);
        let mut calm = AmbientVector::zeros(); calm[10] = 1e-12;
        for _ in 0..200 { lattice.process(0, &calm, Valence::HIGH); }
        let recovered = lattice.effective_purify_period(0);
        assert!(recovered > tight);
        assert!(lattice.zones[0].grief_absorbed > 100.0);
        assert!(lattice.zones[0].stress_ema < lattice.zones[0].grief_absorbed);
    }
