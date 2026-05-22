// ... existing code in SovereignHealthMonitor ...

    /// Evaluates current sovereign state through a chosen mercy gate level.
    /// Now considers more context from current metrics.
    pub fn evaluate_current_state_mercy(
        &self,
        level: mercy_gating::MercyGateLevel,
    ) -> mercy_gating::MercyVerdict {
        // Build richer context from multiple metrics
        let context_score = (
            self.metrics.mercy_compliance * 0.5 +
            self.metrics.valence_level * 0.3 +
            0.2 // base stability factor
        ).min(1.0);

        match level {
            mercy_gating::MercyGateLevel::Foundational => {
                if context_score >= 0.80 {
                    mercy_gating::MercyVerdict::Passed { overall_score: context_score }
                } else if context_score >= 0.65 {
                    mercy_gating::MercyVerdict::Mitigated {
                        overall_score: context_score,
                        notes: vec!["Foundational mercy evaluation on current state".to_string()],
                    }
                } else {
                    mercy_gating::MercyVerdict::RequiresCouncilReview
                }
            }
            mercy_gating::MercyGateLevel::Operational => {
                if context_score >= 0.82 {
                    mercy_gating::MercyVerdict::Passed { overall_score: context_score }
                } else if context_score >= 0.68 {
                    mercy_gating::MercyVerdict::Mitigated {
                        overall_score: context_score,
                        notes: vec!["Operational mercy evaluation on current state".to_string()],
                    }
                } else {
                    mercy_gating::MercyVerdict::RequiresCouncilReview
                }
            }
            mercy_gating::MercyGateLevel::Integrative => {
                if context_score >= 0.85 {
                    mercy_gating::MercyVerdict::Passed { overall_score: context_score }
                } else if context_score >= 0.72 {
                    mercy_gating::MercyVerdict::Mitigated {
                        overall_score: context_score,
                        notes: vec!["Integrative mercy evaluation on current state".to_string()],
                    }
                } else {
                    mercy_gating::MercyVerdict::RequiresCouncilReview
                }
            }
        }
    }

// ... existing code ...