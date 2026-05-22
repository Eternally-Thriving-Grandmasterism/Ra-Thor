// ... existing code ...

    pub fn run_sovereign_check(&mut self) -> SovereignHealthMetrics {
        self.metrics.valence_level = (self.metrics.valence_level + 0.001).min(0.999999);
        self.metrics.mercy_compliance = (self.metrics.mercy_compliance + 0.005).min(1.0);

        // Example integration point for future mercy-based health adjustments
        // if let Some(mercy_impact) = ... { adjust metrics }

        self.metrics
    }

// ... existing code ...