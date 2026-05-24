/// Advanced Orchestrator
///
/// Test philosophy: We test for confidence in orchestration behavior,
/// observability side-effects, accessibility quality signals, and CI gate decisions.
/// Tests should be readable, meaningful, and resistant to brittle changes.

// ... (implementation)

#[cfg(test)]
mod thoughtful_tests {
    use super::*;
    use crate::orchestration::OrchestrationReport;

    #[test]
    fn test_full_orchestration_flow_produces_report() {
        let orchestrator = AdvancedOrchestrator::new();
        let result = orchestrator.orchestrate("Create an accessible landing page");

        let report = OrchestrationReport::from(&result);

        // Basic sanity: report should reflect the result
        assert_eq!(report.success, result.success);
        assert_eq!(report.attempts, result.attempts_used);
    }

    #[test]
    fn test_orchestration_with_multiple_attempts_records_attempts() {
        let mut orchestrator = AdvancedOrchestrator::new();
        orchestrator = orchestrator.with_max_attempts(3);

        let result = orchestrator.orchestrate("Test multiple refinement attempts");

        assert!(result.attempts_used <= 3);
    }
}
