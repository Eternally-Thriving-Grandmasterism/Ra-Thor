//! Integration test for the closed self-evolution loop
//! Audit → Decide → Improve → Verify

use ra_thor_meta_intelligence::{SelfImprovementOrchestrator, AuditSignal};
use ra_thor_monorepo_auditor::MercyMetrics;
use plasticity_engine_v2::SafePlasticityApplicator;

#[test]
fn test_full_self_evolution_cycle_basic_flow() {
    let mut orchestrator = SelfImprovementOrchestrator::new();
    let _applicator = SafePlasticityApplicator::new();

    let audit_signals = vec![
        AuditSignal::new("high_drift".to_string(), 0.8, MercyMetrics::default())
    ];

    let proposals = orchestrator.run_self_evolution_cycle(&audit_signals);
    println!("Generated {} proposals", proposals.len());

    if let Some(p) = proposals.first() {
        let _ = orchestrator.apply_improvement_proposal(p);
    }
}