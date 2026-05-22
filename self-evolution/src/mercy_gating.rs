//! Parallel Progress: More Tests + Self-Referential Improvement

// Improved self-referential logic with clearer weighting and documentation
pub fn self_referential_mercy_evaluation(
    proposal_score: f64,
    current_system_coherence: f64,
    current_mercy_compliance: f64,
) -> MercyVerdict {
    // Weighted combination favoring proposal quality while respecting system state
    let combined = (proposal_score * 0.55)
        + (current_system_coherence * 0.25)
        + (current_mercy_compliance * 0.20);

    if combined >= 0.915 {
        MercyVerdict::Passed { overall_score: combined }
    } else if combined >= 0.81 {
        MercyVerdict::Mitigated {
            overall_score: combined,
            notes: vec!["Self-referential evaluation passed with system awareness".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// Additional MaatKpi edge case tests
#[cfg(test)]
mod more_maat_kpi_tests {
    use super::*;

    #[test]
    fn test_maat_kpi_extreme_imbalance() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 1.0);
        kpi.set_score(MaatDimension::Balance, 0.0);
        kpi.set_score(MaatDimension::Justice, 1.0);
        kpi.set_score(MaatDimension::Order, 0.0);

        let score = kpi.overall_score();
        assert!(score >= 0.0 && score <= 1.0);
        // Coherence should be low due to high variance
        assert!(kpi.coherence_score() < 0.5);
    }
}

// ... existing code ...