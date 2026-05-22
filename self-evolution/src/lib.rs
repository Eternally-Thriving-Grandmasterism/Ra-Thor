// ... existing code in SovereignHealthMonitor ...

    /// Handles cases where mercy evaluation returns RequiresCouncilReview.
    /// This is the starting point for PATSAGi Council triggering.
    fn handle_requires_council_review(&mut self) {
        // Increment a counter or log the event
        // In the future, this can trigger actual PATSAGi Council review
        println!("[SovereignHealthMonitor] RequiresCouncilReview triggered - potential PATSAGi review point");

        // Future enhancement:
        // - Queue for PATSAGi Council
        // - Notify relevant councils
        // - Apply additional mercy or coherence checks
    }

    /// Enhanced apply_mercy_verdict with PATSAGi triggering hook
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

                // Trigger PATSAGi review handling
                self.handle_requires_council_review();
            }
            mercy_gating::MercyVerdict::Blocked { .. } => {
                self.metrics.mercy_compliance = (self.metrics.mercy_compliance * 0.7).max(0.4);
            }
        }
    }

// ... existing code ...