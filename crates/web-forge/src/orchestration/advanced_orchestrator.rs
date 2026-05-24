/// Advanced Orchestrator
///
/// With expanded high-quality test coverage.

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::semantic_planning::SemanticPlanningStrategy;
use crate::validation::{calculate_wcag_aa_score, HtmlValidator};
use crate::observability;
use std::time::Instant;
use tracing::info_span;

// ... (implementation)

#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_orchestrator_produces_structured_result() {
        let orchestrator = AdvancedOrchestrator::new();
        let result = orchestrator.orchestrate("Create a professional dashboard");

        assert!(result.attempts_used >= 1);
        // Even on failure, we should get a structured result
    }

    #[test]
    fn test_wcag_aa_scoring_produces_valid_output() {
        let html = "<html><body><h1>Hello</h1><img src='test.png' alt='test'></body></html>";
        let score = calculate_wcag_aa_score(html);

        assert!(score.score >= 0.0 && score.score <= 100.0);
        assert!(!score.grade.is_empty());
    }

    #[test]
    fn test_orchestration_with_poor_accessibility_scores_low() {
        let bad_html = "<html><body><img src='bad.png'></body></html>"; // missing alt
        let score = calculate_wcag_aa_score(bad_html);

        assert!(score.score < 90.0); // Should be penalized
    }

    #[test]
    fn test_report_generation_from_result() {
        let result = AdvancedOrchestrationResult {
            success: true,
            attempts_used: 2,
            validation_issues: vec![],
            wcag_aa_score: None,
            ..Default::default()
        };

        let report = crate::orchestration::OrchestrationReport::from(&result);
        assert!(report.success);
        assert_eq!(report.attempts, 2);
    }

    #[test]
    fn test_html_validator_detects_accessibility_issues() {
        let validator = HtmlValidator::new();
        let bad_html = "<html><body><img src='x.png'></body></html>";

        let issues = validator.validate(bad_html);
        assert!(!issues.is_empty());
    }
}
