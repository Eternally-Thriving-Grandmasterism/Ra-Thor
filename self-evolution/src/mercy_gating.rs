//! Self-Referential Mercy Tests

#[cfg(test)]
mod self_referential_mercy_tests {
    use super::*;

    #[test]
    fn test_self_referential_high_scores_passed() {
        let verdict = self_referential_mercy_evaluation(0.95, 0.92, 0.90);
        assert!(matches!(verdict, MercyVerdict::Passed { .. }));
    }

    #[test]
    fn test_self_referential_medium_scores_mitigated() {
        let verdict = self_referential_mercy_evaluation(0.82, 0.80, 0.78);
        assert!(matches!(verdict, MercyVerdict::Mitigated { .. }));
    }

    #[test]
    fn test_self_referential_low_scores_requires_review() {
        let verdict = self_referential_mercy_evaluation(0.65, 0.70, 0.68);
        assert!(matches!(verdict, MercyVerdict::RequiresCouncilReview));
    }

    #[test]
    fn test_self_referential_edge_high_proposal_low_coherence() {
        let verdict = self_referential_mercy_evaluation(0.93, 0.65, 0.70);
        // Should likely be Mitigated or RequiresCouncilReview due to low coherence
        assert!(!matches!(verdict, MercyVerdict::Passed { .. }));
    }
}

// ... existing code ...