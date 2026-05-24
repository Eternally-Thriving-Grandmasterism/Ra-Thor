/// Advanced Orchestrator
///
/// With integration test coverage for quality gates.

// ... (implementation)

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::orchestration::OrchestrationReport;
    use crate::validation::WcagAaScore;

    fn make_result(success: bool, score: Option<f32>) -> AdvancedOrchestrationResult {
        AdvancedOrchestrationResult {
            success,
            attempts_used: 1,
            validation_issues: vec![],
            wcag_aa_score: score.map(|s| WcagAaScore {
                score: s,
                issues: vec![],
                grade: if s >= 80.0 { "B".to_string() } else { "C".to_string() },
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_quality_gate_passes_with_good_score() {
        let result = make_result(true, Some(88.0));
        let report = OrchestrationReport::from(&result);

        assert!(report.passes_ci_gate(80.0));
    }

    #[test]
    fn test_quality_gate_fails_on_low_score() {
        let result = make_result(true, Some(65.0));
        let report = OrchestrationReport::from(&result);

        assert!(!report.passes_ci_gate(80.0));
    }

    #[test]
    fn test_quality_gate_fails_on_orchestration_failure() {
        let result = make_result(false, Some(90.0));
        let report = OrchestrationReport::from(&result);

        assert!(!report.passes_ci_gate(70.0));
    }

    #[test]
    fn test_quality_gate_handles_missing_score() {
        let result = make_result(true, None);
        let report = OrchestrationReport::from(&result);

        // When no score is present, gate should still consider success
        assert!(report.passes_ci_gate(0.0));
    }
}
