// ... existing code in SovereignHealthMonitor ...

    /// Requests an epigenetic blessing with synergy from the SelfEvolutionBlessing mercy gate
    pub fn request_epigenetic_blessing_with_mercy(
        &mut self,
        reason: &str,
        base_mercy_score: f64,
    ) -> BlessingResult {
        let current_level = self.metrics.epigenetic_blessing_level;
        let success_rate = if self.blessing_history.is_empty() {
            0.5
        } else {
            let successes = self.blessing_history.iter().filter(|b| b.success).count();
            successes as f64 / self.blessing_history.len() as f64
        };

        // Use the new synergy function from mercy_gating
        let verdict = mercy_gating::evaluate_self_evolution_blessing(
            base_mercy_score,
            current_level,
            success_rate,
        );

        let blessing_score = self.calculate_blessing_score(reason);

        // Combine mercy verdict with traditional blessing scoring
        let final_score = match verdict {
            mercy_gating::MercyVerdict::Passed { overall_score } => (blessing_score + overall_score) / 2.0,
            mercy_gating::MercyVerdict::Mitigated { overall_score, .. } => (blessing_score + overall_score * 0.9) / 2.0,
            mercy_gating::MercyVerdict::RequiresCouncilReview => blessing_score * 0.7,
            mercy_gating::MercyVerdict::Blocked { .. } => 0.1,
        };

        let tier = calculate_blessing_tier(final_score);
        let success = final_score > 0.6;

        let result = BlessingResult {
            tier,
            score: final_score,
            success,
            reason: reason.to_string(),
        };

        self.blessing_history.push(result.clone());

        if success {
            self.metrics.epigenetic_blessing_level =
                (self.metrics.epigenetic_blessing_level + 0.03).min(0.95);
        }

        result
    }

// ... existing code ...