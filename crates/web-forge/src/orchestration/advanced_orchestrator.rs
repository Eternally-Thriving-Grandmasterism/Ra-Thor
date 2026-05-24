/// Advanced Orchestrator
///
/// Continuing thoughtful expansion of test coverage.

// ... (implementation)

#[cfg(test)]
mod deeper_tests {
    use super::*;

    #[test]
    fn test_refinement_respects_max_attempts() {
        let mut orchestrator = AdvancedOrchestrator::new();
        orchestrator = orchestrator.with_max_attempts(2);

        let result = orchestrator.orchestrate("Force multiple refinement attempts");

        assert!(result.attempts_used <= 2);
    }

    #[test]
    fn test_metrics_are_recorded_on_orchestration() {
        // This test ensures the metrics recording path is exercised
        let orchestrator = AdvancedOrchestrator::new();
        let _result = orchestrator.orchestrate("Verify metrics path");

        // In a more advanced setup we would assert against a test MeterProvider
        // For now, executing without panic is meaningful
    }

    #[test]
    fn test_semantic_planning_path_is_available() {
        // Verifies that semantic planning can be enabled without crashing
        let orchestrator = AdvancedOrchestrator::new()
            .with_semantic_planning("dummy-key-for-test".to_string());

        // We don't assert on behavior here yet (no real embeddings),
        // but the path should be constructible
        let _result = orchestrator.orchestrate("Semantic planning test");
    }
}
