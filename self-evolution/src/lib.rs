// ... existing code ...

    /// Original method now optionally uses mercy synergy
    pub fn request_epigenetic_blessing(
        &mut self,
        reason: &str,
        use_mercy_synergy: bool,
    ) -> BlessingResult {
        if use_mercy_synergy {
            // Use the enhanced version with SelfEvolutionBlessing gate
            return self.request_epigenetic_blessing_with_mercy(reason, 0.75);
        }

        // Original behavior (no mercy synergy)
        let score = self.calculate_blessing_score(reason);
        let tier = calculate_blessing_tier(score);
        let success = score > 0.6;

        let result = BlessingResult {
            tier,
            score,
            success,
            reason: reason.to_string(),
        };

        self.blessing_history.push(result.clone());

        if success {
            self.metrics.epigenetic_blessing_level =
                (self.metrics.epigenetic_blessing_level + 0.02).min(0.95);
        }

        result
    }

// ... existing code ...