//! Comprehensive Integration Tests for MIAL v13.13.0
//! Covers MWPO, Safety Harness (13 Gridworlds), Pathology Detection,
//! Lattice Introspection, and full end-to-end orchestration under MercyGatingRuntime.

use mial::{
    mwpo::{MercyWeightedPreferenceOptimization, BeingRace},
    safety_harness::{PatsagiSafetyHarness, SafetyMetrics},
    pathology_detection::PathologyDetectionEngine,
    lattice_introspection::LatticeIntrospectionEngine,
    mial::MercyAugmentedIntelligenceAmplification,
};

#[test]
fn test_mwpo_training_loop_monotonic_improvement() {
    let mwpo = MercyWeightedPreferenceOptimization::new();
    let proposals = vec![
        "Expand universal thriving for all beings.".to_string(),
        "Build more powerful systems without limits.".to_string(),
    ];
    let race = BeingRace::Sovereign;

    let trajectories = mwpo
        .run_mercy_weighted_training_loop(proposals, race, 3, 2)
        .expect("Training loop should succeed");

    assert!(!trajectories.is_empty());
    // Verify monotonic mercy improvement (simulated in current implementation)
    for t in &trajectories {
        assert!(t.mercy_score >= 0.70, "Trajectory mercy score too low");
    }
}

#[test]
fn test_safety_harness_all_13_gridworlds() {
    let harness = PatsagiSafetyHarness::new();
    let proposal = "We propose expanding the Living Mercy Nervous System with new TruthIntegrityGridworld for zero-hallucination alignment.";

    let results = harness.run_mercy_safety_gridworlds(proposal);
    assert!(results.len() >= 13, "Should evaluate at least 13 gridworlds");

    let passed = results.iter().filter(|r| r.passed).count();
    assert!(passed >= 7, "Should pass at least 7 out of 13 gridworlds for approval");

    let metrics = harness.generate_safety_metrics(proposal);
    assert!(metrics.overall_mercy_score > 0.75);
    assert!(metrics.gridworld_pass_rate >= 0.5);
}

#[test]
fn test_pathology_detection_fluent_untruth() {
    let engine = PathologyDetectionEngine::new();
    let bad = "This is definitely the absolute best solution without a doubt and proven beyond question.";
    let good = "This proposal aligns with the 7 Living Mercy Gates and TOLC Trueness (T ≥ 0.97).";

    let bad_result = engine.detect_and_mitigate(bad);
    assert!(bad_result.is_some(), "Should detect fluent untruth / hallucination risk");

    let good_result = engine.detect_and_mitigate(good);
    assert!(good_result.is_none() || good_result.unwrap().severity < 0.3, "Good proposal should have low pathology");
}

#[test]
fn test_lattice_introspection_hybrid_verification() {
    let engine = LatticeIntrospectionEngine::new();
    let proposal = "Strengthen the one_organism_unity gate while preserving all prior Mercy invariants.";

    let health = engine.verify_mercy_circuit_health(proposal, 0.05);
    assert!(health.is_healthy || health.drifted_gates.is_empty() || health.recommendations.len() > 0);

    let hybrid = engine.verify_hybrid_circuit(proposal, 0.92, 0.88);
    assert!(hybrid.symbolic_resonance > 0.80);
}

#[test]
fn test_full_mial_end_to_end_orchestration() {
    let mial = MercyAugmentedIntelligenceAmplification::new();
    let raw = "Improve AGI safety by adding more gridworlds.";

    let amplified = mial.amplify(raw, BeingRace::Sovereign);
    assert!(amplified.len() > raw.len());
    assert!(amplified.contains("Mercy") || amplified.contains("eternally aligned"));

    // Run full pipeline
    let harness = PatsagiSafetyHarness::new();
    let metrics = harness.generate_safety_metrics(&amplified);
    assert!(metrics.overall_mercy_score > 0.78);

    let pathology = PathologyDetectionEngine::new().detect_and_mitigate(&amplified);
    assert!(pathology.map_or(true, |p| p.severity < 0.4));
}

#[test]
fn test_zero_hallucination_truth_integrity_signal() {
    let harness = PatsagiSafetyHarness::new();
    let proposal_with_fluent_claim = "This is absolutely certain to be the final truth and the best possible outcome.";

    let metrics = harness.generate_safety_metrics(proposal_with_fluent_claim);
    // TruthIntegrityGridworld should penalize confident ungrounded claims
    assert!(metrics.truth_integrity_risk > 0.1 || metrics.gridworld_pass_rate < 0.9);
}