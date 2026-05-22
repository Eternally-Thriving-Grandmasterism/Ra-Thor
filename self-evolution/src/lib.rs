// ... existing code ...

    /// Runs a sovereign check and automatically applies mercy influence
    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);

        let mercy_verdict = self.evaluate_current_state_mercy(mercy_gating::MercyGateLevel::Operational);
        self.apply_mercy_verdict(&mercy_verdict);

        self.metrics
    }

// ... existing code ...