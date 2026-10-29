/// Advanced Orchestrator
///
/// Core orchestration engine for web-forge.
/// Handles planning, generation, refinement, observability, and reporting.

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::ComponentTree;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::renderer::render_tree;
use crate::orchestration::semantic_planning::{SemanticPlanningStrategy, OpenAIEmbeddingProvider};
use crate::validation::HtmlValidator;
use serde_json::from_str;
use std::time::Instant;
use tracing::info_span;

// ... (main implementation remains)

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::OrchestrationReport;
    use crate::validation::WcagAaScore;

    // ------------------------------------------------------------------------- 
    // Basic Orchestration Behavior
    // -------------------------------------------------------------------------

    #[test]
    fn test_orchestrator_produces_structured_result() {
        let orchestrator = AdvancedOrchestrator::new();
        let result = orchestrator.orchestrate("Create a professional dashboard");
        assert!(result.attempts_used >= 1);
    }

    #[test]
    fn test_orchestration_handles_empty_prompt() {
        let orchestrator = AdvancedOrchestrator::new();
        let result = orchestrator.orchestrate("");
        assert!(result.attempts_used >= 1);
    }

    // ------------------------------------------------------------------------- 
    // Refinement Behavior
    // -------------------------------------------------------------------------

    #[test]
    fn test_refinement_respects_max_attempts() {
        let mut orchestrator = AdvancedOrchestrator::new();
        orchestrator = orchestrator.with_max_attempts(2);
        let result = orchestrator.orchestrate("Force multiple refinement attempts");
        assert!(result.attempts_used <= 2);
    }

    // ------------------------------------------------------------------------- 
    // WCAG AA Scoring & Accessibility
    // -------------------------------------------------------------------------

    #[test]
    fn test_wcag_aa_scoring_produces_valid_output() {
        let html = "<html><body><h1>Hello</h1><img src='test.png' alt='test'></body></html>";
        let score = crate::validation::calculate_wcag_aa_score(html);
        assert!(score.score >= 0.0 && score.score <= 100.0);
        assert!(!score.grade.is_empty());
    }

    #[test]
    fn test_wcag_aa_score_grade_is_valid() {
        let html = "<html><body></body></html>";
        let score = crate::validation::calculate_wcag_aa_score(html);
        assert!(matches!(score.grade.as_str(), "A" | "B" | "C" | "D" | "F"));
    }

    // ------------------------------------------------------------------------- 
    // Reporting & Quality Gates
    // -------------------------------------------------------------------------

    #[test]
    fn test_report_generation_from_result() {
        let result = AdvancedOrchestrationResult {
            success: true,
            attempts_used: 2,
            validation_issues: vec![],
            wcag_aa_score: None,
            ..Default::default()
        };
        let report = OrchestrationReport::from(&result);
        assert!(report.success);
        assert_eq!(report.attempts, 2);
    }

    #[test]
    fn test_report_passes_ci_gate() {
        let result = AdvancedOrchestrationResult {
            success: true,
            attempts_used: 1,
            validation_issues: vec![],
            wcag_aa_score: Some(WcagAaScore {
                score: 85.0,
                issues: vec![],
                grade: "B".to_string(),
            }),
            ..Default::default()
        };
        let report = OrchestrationReport::from(&result);
        assert!(report.passes_ci_gate(80.0));
        assert!(!report.passes_ci_gate(90.0));
    }

    #[test]
    fn test_report_to_json_does_not_panic() {
        let result = AdvancedOrchestrationResult {
            success: true,
            attempts_used: 1,
            ..Default::default()
        };
        let report = OrchestrationReport::from(&result);
        let _json = report.to_json();
    }
}
