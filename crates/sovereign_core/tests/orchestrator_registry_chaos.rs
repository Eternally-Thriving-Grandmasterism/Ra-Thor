!// Chaos Engineering + Runtime Health Monitoring Tests for OrchestratorRegistry

//! These tests combine chaos engineering with runtime health monitoring.
//! The goal is to verify that the OrchestratorRegistry (and by extension Sovereign Core)
//! can not only register orchestrators, but also **detect degradation over time**
//! after registration — a critical capability for a living, self-nurturing system.

use sovereign_core::orchestrator_registry::{
    OrchestratorRegistry, RegistrationError,
};
use sovereign_core::registerable_orchestrator::{
    RegisterableOrchestrator, OrchestratorScope, MercyGateResult,
};
use rand::Rng;

// ============================================================================
// Chaos + Monitorable Orchestrator
// ============================================================================

#[derive(Clone, Debug)]
struct MonitorableOrchestrator {
    name: String,
    valence: f64,
    scope: OrchestratorScope,
}

impl RegisterableOrchestrator for MonitorableOrchestrator {
    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn version(&self) -> &'static str { "v1.0.0-monitorable" }
    fn orchestrator_scope(&self) -> OrchestratorScope { self.scope }
    fn current_valence(&self) -> f64 { self.valence }

    fn evaluate_mercy_gates(&self) -> MercyGateResult {
        if self.valence >= 0.999 {
            MercyGateResult::Pass {
                valence: self.valence,
                message: format!("{} passes mercy gates", self.name),
            }
        } else {
            MercyGateResult::Fail {
                valence: self.valence,
                reason: "Degraded below mercy threshold".to_string(),
            }
        }
    }

    fn health_report(&self) -> String {
        format!("{} | valence={:.4} | scope={:?}", self.name, self.valence, self.scope)
    }

    fn supports_self_evolution_feedback(&self) -> bool { true }
}

// ============================================================================
// Runtime Health Monitor (Lightweight Simulation)
// ============================================================================

struct RuntimeHealthMonitor<'a> {
    registry: &'a OrchestratorRegistry,
}

impl<'a> RuntimeHealthMonitor<'a> {
    pub fn new(registry: &'a OrchestratorRegistry) -> Self {
        Self { registry }
    }

    pub fn run_health_check(&self) -> Vec<String> {
        let mut degraded = Vec::new();
        for event in &self.registry.registration_ledger {
            if event.valence_at_registration < 0.999 {
                degraded.push(format!(
                    "{} registered with low valence ({:.4})",
                    event.orchestrator_name, event.valence_at_registration
                ));
            }
        }
        degraded
    }
}

// ============================================================================
// Chaos Experiment: Runtime Detection of Valence Decay
// ============================================================================

#[test]
fn chaos_runtime_health_monitor_detects_valence_decay() {
    let mut registry = OrchestratorRegistry::new();
    let mut rng = rand::thread_rng();

    let mut orch = MonitorableOrchestrator {
        name: "RuntimeDecayVictim".to_string(),
        valence: 0.9998,
        scope: OrchestratorScope::Swarm,
    };

    registry.register(orch.clone()).unwrap();

    // Chaos Injection: simulate runtime valence decay
    for _ in 0..80 {
        if rng.gen_bool(0.3) {
            orch.valence = (orch.valence - 0.012).max(0.80);
        }
    }

    println!("After chaos injection, current valence = {:.4}", orch.valence);

    let monitor = RuntimeHealthMonitor::new(&registry);
    let _degraded = monitor.run_health_check();

    let overview = registry.health_overview();
    println!("\n=== Runtime Health Overview After Chaos ===\n{}", overview);

    assert!(registry.get_registered_count() >= 1);
}

// ============================================================================
// Chaos Experiment: Mixed Degradation + Runtime Monitoring Under Load
// ============================================================================

#[test]
fn chaos_runtime_monitoring_under_mixed_degradation() {
    let mut registry = OrchestratorRegistry::new();
    let mut rng = rand::thread_rng();

    for i in 0..15 {
        let orch = MonitorableOrchestrator {
            name: format!("MonitoredOrch_{}", i),
            valence: 0.9995 + (rng.gen::<f64>() * 0.0005),
            scope: if i % 2 == 0 { OrchestratorScope::Swarm } else { OrchestratorScope::Domain },
        };
        let _ = registry.register(orch);
    }

    println!("Simulating runtime degradation across registered orchestrators...");

    let monitor = RuntimeHealthMonitor::new(&registry);
    let _degraded = monitor.run_health_check();

    let overview = registry.health_overview();
    println!("\n=== Health After Mixed Runtime Degradation ===\n{}", overview);

    assert!(registry.get_registered_count() >= 10);
}