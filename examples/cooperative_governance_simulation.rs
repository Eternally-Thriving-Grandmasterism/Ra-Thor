// examples/cooperative_governance_simulation.rs
// Expanded simulation with power concentration and multi-objective scenarios

use lattice_conductor_v14::{LatticeConductorEnhancements, CooperativeGame};
use std::collections::HashSet;

fn main() {
    println!("=== Expanded Cooperative Governance Simulation ===\n");

    // Scenario 1: Multi-objective optimization
    println!("--- Scenario 1: Multi-Objective Optimization ---");
    let participants = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    let char_fn = |s: &HashSet<String>| -> f64 {
        let base = s.len() as f64 * 20.0;
        if s.contains("A") { base + 40.0 } else { base }
    };

    let game = CooperativeGame::new(participants.clone(), char_fn);
    let (best, score) = game.optimize_coalition_multi_objective(3, 0.6, 0.4, 6);
    println!("Best coalition: {:?} (score: {:.2})", best, score);

    // Scenario 2: Power concentration detection
    println!("\n--- Scenario 2: Power Concentration Test ---");
    let dominant_participants = vec!["Dominant".to_string(), "Weak1".to_string(), "Weak2".to_string()];
    let dominant_fn = |s: &HashSet<String>| -> f64 {
        if s.contains("Dominant") { 100.0 } else { 10.0 }
    };

    let (decision, insight) = LatticeConductorEnhancements::submit_to_patsagi_with_game_theory(
        &mut lattice_conductor_v14::DistributedMercyMesh::new(),
        "Power concentration test",
        "Testing Banzhaf escalation",
        dominant_participants,
        dominant_fn,
    );
    println!("Decision: {:?}", decision);
    println!("Insight: {}", insight);

    println!("\n=== Simulation Complete ===");
}
