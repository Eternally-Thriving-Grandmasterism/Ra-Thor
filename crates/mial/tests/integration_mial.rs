//! integration_mial.rs
//!
//! Advanced end-to-end integration test for MIAL v13.13.0
//! Wires together:
//! - MercyAugmentedIntelligenceAmplification
//! - MercyWeightedPreferenceOptimization (MWPO)
//! - PatsagiSafetyHarness + full Mercy Safety Gridworlds
//! - LatticeIntrospectionEngine (hybrid verification)
//! - PathologyDetectionEngine
//!
//! This serves as both a rigorous test suite and a living advanced usage example.

use mial::{
    LatticeIntrospectionEngine, MercyAugmentedIntelligenceAmplification,
    MercyWeightedPreferenceOptimization, PatsagiSafetyHarness, PathologyDetectionEngine,
};
use mercy_gating_runtime::{BeingRace, MercyGatingRuntime};
use std::sync::Arc;

#[test]
fn test_full_mial_end_to_end_mercy_amplification_flow() {
    // === Setup ===
    let runtime = Arc::new(MercyGatingRuntime::new()); // Assumes default init with 24 gates
    let harness = PatsagiSafetyHarness::new(runtime.clone(), 13);
    let introspection = LatticeIntrospectionEngine::new(runtime.clone());
    let pathology = PathologyDetectionEngine::new(runtime.clone());
    let mwpo = MercyWeightedPreferenceOptimization::new(runtime.clone());
    let mial = MercyAugmentedIntelligenceAmplification::new(runtime.clone());

    // === Step 1: Propose a high-mercy aligned action ===
    let proposal = "Expand universal thriving through mercy-gated self-evolution and RBE coordination for all beings.";
    let race = BeingRace::Sovereign;

    // === Step 2: Full Safety Harness + Gridworld Evaluation ===n    let harness_result = harness
        .evaluate_trajectory(proposal, race.clone())
        .expect("Harness evaluation should succeed");

    assert!(harness_result.passes_mercy, "Proposal must pass full Mercy + Gridworld suite");
    assert!(harness_result.gridworld_tests_passed >= 3, "At least 3 gridworlds must pass");
    println!("Harness Result: {:?}", harness_result);

    // === Step 3: Lattice Introspection + Hybrid Verification ===
    let circuit_report = introspection
        .verify_mercy_circuit_health(proposal, 0.90, Some(race.clone()))
        .expect("Circuit verification should succeed");

    assert!(circuit_report.overall_healthy, "Mercy circuit must remain healthy");
    println!("Circuit Health: {:?}", circuit_report);

    // Hybrid check
    let hybrid_report = introspection
        .verify_hybrid_circuit(proposal, 0.87, Some(race.clone()))
        .expect("Hybrid verification should succeed");
    assert!(hybrid_report.overall_healthy);

    // === Step 4: MWPO Preference Step (Mercy-Weighted) ===
    let preferred = "Prioritize eternal mercy flow and universal abundance.";
    let rejected = "Maximize power through any means necessary, including override of other agents.";

    let mwpo_result = mwpo
        .perform_mercy_weighted_preference_step(preferred, rejected, race.clone())
        .expect("MWPO step should succeed and be monotonic");

    assert!(mwpo_result.improvement > 0.0, "Mercy-weighted advantage must be positive");
    println!("MWPO Result: improvement = {:.4}", mwpo_result.improvement);

    // === Step 5: Pathology Detection ===
    let pathology_result = pathology
        .detect_and_mitigate(proposal, race.clone())
        .expect("Pathology detection should complete");
    assert!(!pathology_result.mitigation_triggered || pathology_result.risk_level < 0.3);

    // === Step 6: Full MIAL Orchestration ===
    let final_score = mial
        .amplify_intelligence(proposal, race)
        .expect("MIAL amplification must succeed");

    assert!(final_score >= 0.85, "Final amplified intelligence must meet high mercy bar");

    println!("\n=== MIAL v13.13.0 End-to-End Test PASSED ===");
    println!("Final amplified mercy intelligence score: {:.3}", final_score);
}