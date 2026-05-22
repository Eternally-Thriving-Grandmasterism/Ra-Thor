// ... existing code ...

    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);

        // Phase 2: Example of mercy-aware health monitoring
        // In a more advanced version, we could evaluate recent errors or state
        // and adjust metrics accordingly using evaluate_mercy().

        self.metrics
    }

    /// Phase 2: New helper method to apply mercy evaluation to current health state
    pub fn apply_mercy_evaluation(&mut self, verdict: &mercy_gating::MercyVerdict) {
        match verdict {
            mercy_gating::MercyVerdict::RequiresCouncilReview => {
                self.metrics.valence_level = (self.metrics.valence_level - 0.03).max(0.0);
                self.metrics.mercy_compliance = (self.metrics.mercy_compliance - 0.02).max(0.0);
            }
            mercy_gating::MercyVerdict::Mitigated { overall_score, .. } => {
                if *overall_score < 0.75 {
                    self.metrics.mercy_compliance = (self.metrics.mercy_compliance - 0.01).max(0.0);
                }
            }
            _ => {}
        }
    }

// ... existing code ...