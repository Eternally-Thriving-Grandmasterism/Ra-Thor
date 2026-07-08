    // ==================== v13.5 Refined Critique Tests ====================

    #[test]
    fn test_refined_critique_passes_under_good_conditions() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_success_ema = 0.82;
        orchestrator.meta_evolution_rate = 0.02;

        let (passed, _) = orchestrator.perform_meta_critique(
            "TestCouncil",
            "increase_meta_rate",
            0.5,
            0.95,
        );
        assert!(passed);
    }

    #[test]
    fn test_refined_critique_blocks_on_low_mercy() {
        let orchestrator = SelfEvolutionOrchestrator::new();
        let (passed, msg) = orchestrator.perform_meta_critique(
            "TestCouncil",
            "increase_meta_rate",
            0.8,
            0.85, // low mercy
        );
        assert!(!passed);
        assert!(msg.contains("Low current mercy"));
    }

    #[test]
    fn test_refined_critique_blocks_on_elevated_rate() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_success_ema = 0.82;
        orchestrator.meta_evolution_rate = 0.045; // already high

        let (passed, msg) = orchestrator.perform_meta_critique(
            "TestCouncil",
            "increase_meta_rate",
            0.8,
            0.96,
        );
        assert!(!passed);
        assert!(msg.contains("already elevated"));
    }

    #[test]
    fn test_refined_critique_blocks_on_frequent_recent_changes() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_success_ema = 0.82;

        // Simulate recent meta activity
        for _ in 0..4 {
            orchestrator.evolution_history.push("[v13.5 Council Meta] increased meta_evolution_rate".to_string());
        }

        let (passed, msg) = orchestrator.perform_meta_critique(
            "TestCouncil",
            "increase_meta_rate",
            0.75,
            0.96,
        );
        assert!(!passed);
        assert!(msg.contains("Frequent recent"));
    }

    #[test]
    fn test_refined_critique_high_mercy_override() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_success_ema = 0.82;
        orchestrator.meta_evolution_rate = 0.045;

        // Very high mercy should allow override even with concerns
        let (passed, _) = orchestrator.perform_meta_critique(
            "TestCouncil",
            "increase_meta_rate",
            0.8,
            0.97, // very high mercy
        );
        // Note: In actual flow, override happens in council_voted_meta_rate_adjust, not here
        // This test just checks critique still reports concerns
        assert!(!passed);
    }
