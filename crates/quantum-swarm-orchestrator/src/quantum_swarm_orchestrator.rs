use crate::registerable_orchestrator::{RegisterableOrchestrator, OrchestratorScope};
use crate::mercy::MercyGateResult;

// === RegisterableOrchestrator Implementation (Additive) ===
impl RegisterableOrchestrator for QuantumSwarmOrchestrator {
    fn name(&self) -> &'static str {
        "QuantumSwarmOrchestrator"
    }

    fn version(&self) -> &'static str {
        "v1.0.0-registerable"
    }

    fn orchestrator_scope(&self) -> OrchestratorScope {
        OrchestratorScope::Swarm
    }

    fn current_valence(&self) -> f64 {
        self.state.mercy_valence
    }

    fn evaluate_mercy_gates(&self) -> MercyGateResult {
        if self.current_valence() >= 0.999 {
            MercyGateResult::Pass {
                valence: self.current_valence(),
                message: "Quantum Swarm passes mercy alignment".to_string(),
            }
        } else {
            MercyGateResult::Fail {
                valence: self.current_valence(),
                reason: "Valence below required threshold".to_string(),
            }
        }
    }

    fn health_report(&self) -> String {
        self.swarm_health_summary()
    }

    fn coordination_capabilities(&self) -> Vec<&'static str> {
        vec![
            "parallel_agent_coordination",
            "real_time_mercy_propagation",
            "swarm_health_aggregation",
            "plasticity_engine_integration"
        ]
    }

    fn compatible_with(&self) -> Vec<OrchestratorScope> {
        vec![OrchestratorScope::Sovereign, OrchestratorScope::Meta, OrchestratorScope::Domain]
    }

    fn supports_self_evolution_feedback(&self) -> bool {
        true
    }
}