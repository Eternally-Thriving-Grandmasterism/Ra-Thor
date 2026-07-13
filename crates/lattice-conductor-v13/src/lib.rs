        self.metrics.operations_processed += 1;
    }

    /// Integrate backend-aware self-evolution telemetry emitted by MasterKernel.
    /// Performs real state updates (evolution_level, symbolic_success_ema, mercy_score, coherence)
    /// in a positive, mercy-gated manner. Records to audit_traces for PATSAGi observability.
    pub fn integrate_self_evolution_telemetry(&mut self, telemetry: &SelfEvolutionTelemetry) {
        if telemetry.avg_tu_delta > 0.0 {
            self.state.evolution_level += telemetry.avg_tu_delta * 0.05;
            self.symbolic_success_ema = (self.symbolic_success_ema + telemetry.avg_tu_delta * 0.2).clamp(0.0, 1.5);
            self.state.mercy_score = (self.state.mercy_score + telemetry.avg_tu_delta * 0.1).clamp(0.1, 2.0);
            self.one_organism_coherence = (self.one_organism_coherence + telemetry.avg_tu_delta * 0.05).min(1.5);
        }

        self.audit_traces.push(format!(
            "[SelfEvolutionTelemetry] backend={:?} tu_delta={:.4} adaptation={:.4} w_e: {:.4}→{:.4}",
            telemetry.backend,
            telemetry.avg_tu_delta,
            telemetry.adaptation_rate,
            telemetry.old_w_e,
            telemetry.new_w_e
        ));

        self.metrics.operations_processed += 1;
    }

    #[cfg(feature = "self-proposal")]
    pub fn generate_symbolic_self_proposals(&self) -> Vec<SymbolicSelfProposal> {