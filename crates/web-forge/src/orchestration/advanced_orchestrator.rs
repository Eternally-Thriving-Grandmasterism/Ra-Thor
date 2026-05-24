/// Advanced Orchestrator
///
/// Continuing balanced expansion of tests and documentation.

// ... (implementation)

#[cfg(test)]
mod continued_expansion {
    use super::*;

    #[test]
    fn test_orchestrator_with_high_max_attempts() {
        let mut orchestrator = AdvancedOrchestrator::new();
        orchestrator = orchestrator.with_max_attempts(5);

        let result = orchestrator.orchestrate("High attempt limit test");

        assert!(result.attempts_used <= 5);
    }

    #[test]
    fn test_wcag_scoring_on_minimal_html() {
        let minimal = "<html><body></body></html>";
        let score = crate::validation::calculate_wcag_aa_score(minimal);

        // Minimal HTML should score relatively low
        assert!(score.score < 70.0);
    }
}
