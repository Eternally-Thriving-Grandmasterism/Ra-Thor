// ... existing tests preserved exactly ...

    #[test]
    fn test_metta_symbolic_deliberation_high_valence() {
        let result = metta_symbolic_deliberation("council_deliberation", 1.0);
        assert!(result.contains("truth_distilled"));
        assert!(result.contains("NEXi bridge active"));
    }

    #[test]
    fn test_metta_symbolic_deliberation_low_valence() {
        let result = metta_symbolic_deliberation("evolution_step", 0.5);
        assert!(result.contains("compensated_low_valence"));
    }

    // Proptests for metta_symbolic_deliberation (property-based, valence and input invariants)
    proptest! {
        #[test]
        fn metta_deliberation_preserves_truth_distillation_for_high_valence(
            input in ".{1,50}",
            valence in 0.9999999f64..=1.0f64
        ) {
            let result = metta_symbolic_deliberation(&input, valence);
            prop_assert!(result.contains("truth_distilled") || result.contains("NEXi"));
            prop_assert!(result.len() > input.len());
        }

        #[test]
        fn metta_deliberation_compensates_low_valence(
            input in ".{1,30}",
            valence in 0.0f64..0.9999998f64
        ) {
            let result = metta_symbolic_deliberation(&input, valence);
            prop_assert!(result.contains("compensated_low_valence"));
        }

        #[test]
        fn metta_deliberation_valence_monotonicity(
            input in ".{1,20}"
        ) {
            let high = metta_symbolic_deliberation(&input, 1.0);
            let low = metta_symbolic_deliberation(&input, 0.5);
            prop_assert!(high.contains("truth_distilled"));
            prop_assert!(low.contains("compensated"));
        }
    }
}