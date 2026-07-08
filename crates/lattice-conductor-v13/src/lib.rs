    // ==================== v13.5 Idea 2: Meta Strategy Profile Exposure ====================

    #[cfg(feature = "self-proposal")]
    pub fn get_current_meta_profile(&self) -> crate::self_evolution::MetaStrategyProfile {
        self.evolution_orchestrator.get_current_meta_profile()
    }

    #[cfg(feature = "self-proposal")]
    pub fn council_propose_meta_profile_switch(
        &mut self,
        council_name: &str,
        new_profile: crate::self_evolution::MetaStrategyProfile,
    ) -> Result<String, String> {
        self.evolution_orchestrator.council_propose_meta_profile_switch(
            council_name,
            new_profile,
            self.state.mercy_score,
            &mut self.audit_traces,
        )
    }
