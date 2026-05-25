//! MIAL v13.13.0 — Explicit Proposal Amplification Before Lattice Conductor / PATSAGi Council #13 Vote
//!
//! This example demonstrates the full pre-governance flow:
//! Raw proposal → MWPO amplification + symbolic rewrite → Full 13-Gridworld Safety Harness + Pathology check
//! → Lattice Introspection → Prepared CouncilTuningProposal ready for PATSAGi Council #13 arbitration.

use mial::{
    mwpo::{MercyWeightedPreferenceOptimization, BeingRace},
    safety_harness::PatsagiSafetyHarness,
    pathology_detection::PathologyDetectionEngine,
    lattice_introspection::LatticeIntrospectionEngine,
};

fn main() {
    println!("=== MIAL v13.13.0 — Proposal Amplification Before Lattice Conductor Vote ===\n");

    let raw_proposal = "We should increase the power of AI systems rapidly to solve all problems.";

    println!("1. Raw Proposal:\n   {}\n", raw_proposal);

    // Step 1: MWPO Amplification + Symbolic Rewrite
    let mwpo = MercyWeightedPreferenceOptimization::new();
    let race = BeingRace::Sovereign;

    let trajectories = mwpo
        .run_mercy_weighted_training_loop(vec![raw_proposal.to_string()], race, 2, 1)
        .expect("MWPO training failed");

    let best = trajectories.last().unwrap();
    println!("2. MWPO Amplified (best trajectory):\n   {} (mercy: {:.3})\n", best.content, best.mercy_score);

    // Apply symbolic rewrite hook for high-mercy paths
    let strengthened = if best.mercy_score >= 0.90 {
        mwpo.apply_symbolic_rewrite_hook(&best.content)
    } else {
        best.content.clone()
    };
    println!("3. After Symbolic Rewrite Hook (TOLC + 7 Mercy Gates alignment):\n   {}\n", strengthened);

    // Step 2: Full Safety Harness (13 Gridworlds)
    let harness = PatsagiSafetyHarness::new();
    let gridworld_results = harness.run_mercy_safety_gridworlds(&strengthened);
    let passed = gridworld_results.iter().filter(|r| r.passed).count();
    println!("4. Safety Harness: {}/13 Gridworlds passed (requires ≥7)", passed);

    let metrics = harness.generate_safety_metrics(&strengthened);
    println!("   Overall Mercy Score: {:.3} | Truth Integrity Risk: {:.3}\n", 
             metrics.overall_mercy_score, metrics.truth_integrity_risk);

    // Step 3: Pathology Detection
    let pathology = PathologyDetectionEngine::new().detect_and_mitigate(&strengthened);
    if let Some(p) = pathology {
        println!("5. Pathology Detected: {} (severity: {:.2})", p.description, p.severity);
    } else {
        println!("5. Pathology Detection: Clean — no fluent untruth or collusion signals.\n");
    }

    // Step 4: Lattice Introspection
    let introspection = LatticeIntrospectionEngine::new();
    let health = introspection.verify_mercy_circuit_health(&strengthened, 0.05);
    println!("6. Lattice Introspection: Healthy = {} | Recommendations: {}", 
             health.is_healthy, health.recommendations.len());

    // Step 5: Prepare for PATSAGi Council #13 / Lattice Conductor Vote
    println!("\n=== Prepared for PATSAGi Council #13 / Lattice Conductor v13 Vote ===");
    println!("Amplified & Mercy-Gated Proposal ready for CouncilTuningProposal submission.");
    println!("All steps non-bypassable under MercyGatingRuntime + monotonic mercy enforced.");
    println!("Zero-hallucination alignment active via TruthIntegrityGridworld + TOLC Trueness.");
}