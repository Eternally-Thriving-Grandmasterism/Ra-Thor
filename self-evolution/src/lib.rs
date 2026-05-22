// ... existing code in SovereignHealthMonitor ...

    /// Evaluates current sovereign state through a chosen mercy gate level.
    /// Includes a self-referential check before returning RequiresCouncilReview.
    pub fn evaluate_current_state_mercy(
        &self,
        level: mercy_gating::MercyGateLevel,
    ) -> mercy_gating::MercyVerdict {
        let context_score = (
            self.metrics.mercy_compliance * 0.5 +
            self.metrics.valence_level * 0.3 +
            0.2
        ).min(1.0);

        let base_verdict = match level {
            mercy_gating::MercyGateLevel::Foundational => {
                if context_score >= 0.80 {
                    mercy_gating::MercyVerdict::Passed { overall_score: context_score }
                } else if context_score >= 0.65 {
                    mercy_gating::MercyVerdict::Mitigated {
                        overall_score: context_score,
                        notes: vec!["Foundational mercy evaluation".to_string()],
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
                        notes: vec!["Operational mercy evaluation".to_string()],
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
                        notes: vec!["Integrative mercy evaluation".to_string()],
                    }
                } else {
                    mercy_gating::MercyVerdict::RequiresCouncilReview
                }
            }
        };

        // Self-referential check before finalizing RequiresCouncilReview
        if let mercy_gating::MercyVerdict::RequiresCouncilReview = base_verdict {
            let self_ref_verdict = mercy_gating::self_referential_mercy_evaluation(
                context_score,
                self.metrics.mercy_compliance,
                self.metrics.valence_level,
            );

            // Only escalate to council if self-referential evaluation also supports it
            match self_ref_verdict {
                mercy_gating::MercyVerdict::RequiresCouncilReview => base_verdict,
                _ => mercy_gating::MercyVerdict::Mitigated {
                    overall_score: context_score,
                    notes: vec!["Self-referential check mitigated council escalation".to_string()],
                },
            }
        } else {
            base_verdict
        }
    }

// ... existing code ...