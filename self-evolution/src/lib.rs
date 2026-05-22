// ... existing code in SovereignHealthMonitor ...

    /// Evaluates an evolution proposal through mercy gates
    pub fn evaluate_evolution_proposal(
        &self,
        proposal_score: f64,
    ) -> mercy_gating::MercyVerdict {
        // Use self-referential evaluation for proposals
        mercy_gating::self_referential_mercy_evaluation(
            proposal_score,
            self.metrics.mercy_compliance,
            self.metrics.valence_level,
        )
    }

    /// Apply mercy verdict influence across multiple metrics
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
                // Could later trigger PATSAGi-style review here
            }
            mercy_gating::MercyVerdict::Blocked { .. } => {
                self.metrics.mercy_compliance = (self.metrics.mercy_compliance * 0.7).max(0.4);
            }
        }
    }

// ... existing code ...