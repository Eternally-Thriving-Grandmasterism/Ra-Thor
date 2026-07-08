    // ==================== v13.5: Council Specialization & Decision-Type Weighting (Idea 1) ====================

    #[cfg(feature = "self-proposal")]
    pub fn set_council_weight(&mut self, council_name: &str, weight: f64) {
        self.evolution_orchestrator.set_council_weight(council_name, weight);
    }

    #[cfg(feature = "self-proposal")]
    pub fn set_council_decision_multiplier(&mut self, council_name: &str, decision: crate::self_evolution::MetaDecisionType, multiplier: f64) {
        self.evolution_orchestrator.set_council_decision_multiplier(council_name, decision, multiplier);
    }

    #[cfg(feature = "self-proposal")]
    pub fn council_voted_meta_rate_adjust(
        &mut self,
        council_name: &str,
        proposal_type: &str,
        strength: f64,
    ) -> Result<String, String> {
        self.evolution_orchestrator.council_voted_meta_rate_adjust(
            council_name,
            proposal_type,
            strength,
            self.state.mercy_score,
            &mut self.audit_traces,
        )
    }
