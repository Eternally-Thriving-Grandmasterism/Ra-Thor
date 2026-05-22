//! Ra-Thor Guided Parallel Work: Self-Referential + TolcFidelity + Tests

/// Enhanced self-referential evaluation connected to health monitor concepts
pub fn self_referential_mercy_evaluation(
    proposal_score: f64,
    current_system_coherence: f64,
    current_mercy_compliance: f64,
) -> MercyVerdict {
    let combined = (proposal_score * 0.5)
        + (current_system_coherence * 0.3)
        + (current_mercy_compliance * 0.2);

    if combined >= 0.91 {
        MercyVerdict::Passed { overall_score: combined }
    } else if combined >= 0.80 {
        MercyVerdict::Mitigated {
            overall_score: combined,
            notes: vec!["Self-referential: Proposal aligns with current system mercy state".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

/// Further deepened TolcFidelity with self-referential flavor
pub fn evaluate_tolc_fidelity_with_context(
    base_score: f64,
    current_mercy_compliance: f64,
) -> MercyVerdict {
    let adjusted = base_score + (current_mercy_compliance * 0.09);

    if adjusted >= 0.94 {
        MercyVerdict::Passed { overall_score: adjusted }
    } else if adjusted >= 0.84 {
        MercyVerdict::Mitigated {
            overall_score: adjusted,
            notes: vec![
                "TOLC Fidelity: High origin coherence + self-state alignment".to_string()
            ],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// Added tests for new self-referential logic
#[cfg(test)]
mod self_referential_tests {
    use super::*;

    #[test]
    fn test_self_referential_evaluation() {
        let verdict = self_referential_mercy_evaluation(0.92, 0.88, 0.85);
        assert!(matches!(verdict, MercyVerdict::Passed { .. }) || matches!(verdict, MercyVerdict::Mitigated { .. }));
    }
}

// ... existing code ...