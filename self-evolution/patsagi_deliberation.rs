//! Ra-Thor™ PATSAGi Deliberation & Message-Passing Module v0.2
//! Expanded with multi-round deliberation and stronger consensus logic
//! 100% Proprietary — AG-SML v1.0

// ... (previous code) ...

impl DeliberationSession {
    /// Run multiple deliberation rounds with message accumulation
    pub fn run_multi_round_deliberation(&mut self, rounds: u32) -> f64 {
        let mut final_consensus = 0.5;

        for _ in 0..rounds {
            final_consensus = self.calculate_consensus();

            // Simple stabilization: reduce extreme swings in later rounds
            if self.messages.len() > 5 {
                final_consensus = final_consensus * 0.7 + 0.5 * 0.3;
            }

            self.final_consensus = Some(final_consensus);
        }

        final_consensus
    }

    /// Get summary of current deliberation state
    pub fn get_summary(&self) -> String {
        format!(
            "Topic: {} | Messages: {} | Endorsement Strength: {:.2} | Concern Strength: {:.2} | Consensus: {:.2}",
            self.topic,
            self.messages.len(),
            self.endorsements.values().sum::<f64>(),
            self.concerns.values().sum::<f64>(),
            self.final_consensus.unwrap_or(0.5)
        )
    }
}