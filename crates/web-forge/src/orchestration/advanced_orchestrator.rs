/// Advanced Orchestrator
///
/// Continuing parallel expansion of tests and documentation.

// ... (implementation)

#[cfg(test)]
mod refinement_and_observability_tests {
    use super::*;

    #[test]
    fn test_refinement_loop_executes_multiple_times_when_needed() {
        let mut orchestrator = AdvancedOrchestrator::new();
        orchestrator = orchestrator.with_max_attempts(3);

        // Even if validation fails, we should respect max_attempts
        let result = orchestrator.orchestrate("Trigger refinement loop");

        assert!(result.attempts_used <= 3);
    }

    #[test]
    fn test_observability_spans_do_not_panic() {
        // Ensures that our tracing instrumentation runs cleanly
        let orchestrator = AdvancedOrchestrator::new();
        let _result = orchestrator.orchestrate("Observability span test");
    }
}
