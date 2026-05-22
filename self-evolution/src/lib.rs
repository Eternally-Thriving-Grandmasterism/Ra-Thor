// ... existing code in SovereignHealthMonitor ...

    /// Runs a full sovereign check with integrated mercy evaluation
    pub fn run_sovereign_check_with_mercy(&mut self) -> SovereignHealthMetrics {
        // Run base check
        let mut metrics = self.run_sovereign_check();

        // Create a simple mercy evaluation on current state
        // For now we use a neutral base score; in future this can be more contextual
        let base_mercy_score = 0.82;

        let mercy_verdict = mercy_gating::evaluate_with_cross_layer(
            base_mercy_score,
            None,
            None,
            mercy_gating::MercyGateLevel::Integrative,
        );

        // Apply mercy influence to metrics
        match mercy_verdict {
            mercy_gating::MercyVerdict::Passed { overall_score } => {
                metrics.mercy_compliance = (metrics.mercy_compliance + overall_score * 0.05).min(0.999);
                metrics.valence_level = (metrics.valence_level + 0.02).min(0.999);
            }
            mercy_gating::MercyVerdict::Mitigated { overall_score, .. } => {
                metrics.mercy_compliance = (metrics.mercy_compliance + overall_score * 0.03).min(0.999);
            }
            mercy_gating::MercyVerdict::RequiresCouncilReview => {
                metrics.mercy_compliance = (metrics.mercy_compliance - 0.02).max(0.5);
            }
            _ => {}
        }

        metrics
    }

    /// Evaluates current sovereign state through a chosen mercy gate level
    pub fn evaluate_current_state_mercy(
        &self,
        level: mercy_gating::MercyGateLevel,
    ) -> mercy_gating::MercyVerdict {
        // In a more advanced version, we would build a richer context from metrics
        let context_score = (self.metrics.mercy_compliance + self.metrics.valence_level) / 2.0;
        // Simple evaluation for now
        if context_score >= 0.85 {
            mercy_gating::MercyVerdict::Passed { overall_score: context_score }
        } else if context_score >= 0.70 {
            mercy_gating::MercyVerdict::Mitigated {
                overall_score: context_score,
                notes: vec!["Current sovereign state evaluated through mercy".to_string()],
            }
        } else {
            mercy_gating::MercyVerdict::RequiresCouncilReview
        }
    }

// ... existing code ...