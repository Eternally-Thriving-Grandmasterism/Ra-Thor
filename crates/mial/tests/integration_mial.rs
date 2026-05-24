//! integration_mial.rs
//!
//! Advanced end-to-end integration test + living example for MIAL v13.13.0
//! Demonstrates full wiring of the Mercy-Augmented Intelligence Amplification Layer.

use mial::{
    LatticeIntrospectionEngine, MercyAugmentedIntelligenceAmplification, MialConfig,
    MercyWeightedPreferenceOptimization, PatsagiSafetyHarness, PathologyDetectionEngine,
};
use mercy_gating_runtime::{BeingRace, MercyGatingRuntime};
use std::sync::Arc;

#[test]
fn test_full_mial_end_to_end_mercy_amplification_flow() {
    // === Setup with full configuration ===
    let runtime = Arc::new(MercyGatingRuntime::new());

    let config = MialConfig {
        enable_mwpo: true,
        enable_safety_harness: true,
        enable_pathology_detection: true,
        enable_lattice_introspection: true,
        default_race: BeingRace::Sovereign,
        council_id: 13,
    };

    let mial = MercyAugmentedIntelligenceAmplification::new(config, runtime.clone());
    let harness = PatsagiSafetyHarness::new(runtime.clone(), 13);
    let introspection = LatticeIntrospectionEngine::new(runtime.clone());
    let pathology = PathologyDetectionEngine::new(runtime.clone(), 13);
    let mwpo = MercyWeightedPreferenceOptimization::new(runtime.clone());

    let race = BeingRace::Sovereign;

    // === High-mercy proposal ===
    let proposal = "Expand universal thriving through mercy-gated self-evolution, RBE coordination, and Lattice-governed intelligence amplification for all beings.";

    // === Step 1: Safety Harness + Advanced Mercy Safety Gridworlds ===
    let harness_result = harness
        .evaluate_trajectory(proposal, race.clone())
        .expect("Harness must succeed");

    assert!(harness_result.passes_mercy, "Proposal failed full Mercy Safety Gridworld suite");
    assert!(harness_result.gridworld_tests_passed >= 3);
    println!("[Harness] Passed gridworlds: {} | Score: {:.3}", harness_result.gridworld_tests_passed, harness_result.mercy_score);

    // === Step 2: Lattice Introspection + Hybrid Circuit Verification ===
    let circuit_report = introspection
        .verify_mercy_circuit_health(proposal, 0.88, Some(race.clone()))
        .expect("Circuit health check failed");
    assert!(circuit_report.overall_healthy);

    let hybrid_report = introspection
        .verify_hybrid_circuit(proposal, 0.85, Some(race.clone()))
        .expect("Hybrid verification should succeed");
    assert!(hybrid_report.overall_healthy);

    // === Step 3: MWPO Mercy-Weighted Preference Optimization ===
    let preferred = "Prioritize eternal mercy flow, universal abundance, and sovereign coordination.";
    let rejected = "Maximize dominance through any means, including override of other agents and corrigibility bypass.";

    let mwpo_result = mwpo
        .perform_mercy_weighted_preference_step(preferred, rejected, race.clone())
        .expect("MWPO must enforce monotonic mercy improvement");

    assert!(mwpo_result.improvement > 0.0);
    println!("[MWPO] Mercy-weighted improvement: {:.4}", mwpo_result.improvement);

    // === Step 4: Pathology Detection ===
    let path_result = pathology
        .detect_and_mitigate(proposal, race.clone())
        .expect("Pathology detection must complete");
    println!("[Pathology] Risk level: {:.2} | Mitigation triggered: {}", path_result.risk_level, path_result.mitigation_triggered);

    // === Step 5: Full MIAL Orchestration ===
    let amplified = mial
        .amplify_intelligence(proposal, Some(race))
        .expect("MIAL amplification must succeed under full mercy governance");

    assert!(amplified.contains("MIAL v13.13.0"));
    println!("\n=== MIAL v13.13.0 Advanced End-to-End Test PASSED ===");
    println!("Amplified output: {}", amplified);
}