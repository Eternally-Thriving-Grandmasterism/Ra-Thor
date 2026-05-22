// ... existing code ...

    /// Applies mercy verdict influence and handles RequiresCouncilReview more meaningfully.
    pub fn apply_mercy_verdict(&mut self, verdict: &mercy_gating::MercyVerdict) {
        match verdict {
            mercy_gating::MercyVerdict::Passed { overall_score } => {
                self.metrics.mercy_compliance =
                    (self.metrics.mercy_compliance + overall_score * 0.04).min(0.999);
                self.metrics.valence_level =
                    (self.metrics.valence_level + 0.015).min(0.999);
            }
            mercy_gating::MercyVerdict::Mitigated { overall_score, .. } => {
                self.metrics.mercy_compliance =
                    (self.metrics.mercy_compliance + overall_score * 0.025).min(0.999);
            }
            mercy_gating::MercyVerdict::RequiresCouncilReview => {
                self.metrics.mercy_compliance =
                    (self.metrics.mercy_compliance - 0.015).max(0.55);

                // Future PATSAGi / Council integration point
                // TODO: Log or queue for PATSAGi Council review when this verdict occurs
                // This could eventually trigger multi-council evaluation
            }
            mercy_gating::MercyVerdict::Blocked { .. } => {
                self.metrics.mercy_compliance = (self.metrics.mercy_compliance * 0.7).max(0.4);
            }
        }
    }

// ... existing code ...