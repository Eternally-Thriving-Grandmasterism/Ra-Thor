// ... existing code ...

    /// Enhanced run_sovereign_check that always includes mercy evaluation
    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);

        // Apply light mercy influence on every check
        let mercy_verdict = self.evaluate_current_state_mercy(mercy_gating::MercyGateLevel::Operational);

        match mercy_verdict {
            mercy_gating::MercyVerdict::RequiresCouncilReview => {
                self.metrics.mercy_compliance = (self.metrics.mercy_compliance - 0.01).max(0.6);
            }
            _ => {}
        }

        self.metrics
    }

// ... existing code ...