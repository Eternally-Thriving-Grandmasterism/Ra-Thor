// ... existing code ...

    /// Core sovereign health check (light mercy influence included)
    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);

        // Light mercy evaluation on every check
        let mercy_verdict = self.evaluate_current_state_mercy(mercy_gating::MercyGateLevel::Operational);

        if let mercy_gating::MercyVerdict::RequiresCouncilReview = mercy_verdict {
            self.metrics.mercy_compliance = (self.metrics.mercy_compliance - 0.01).max(0.6);
        }

        self.metrics
    }

// ... existing code ...