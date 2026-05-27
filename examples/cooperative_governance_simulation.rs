// examples/cooperative_governance_simulation.rs
// Demonstrates multi-objective Shapley optimization + intelligent PATSAGi integration

use lattice_conductor_v14::{LatticeConductorEnhancements, CooperativeGame};
use std::collections::HashSet;

fn main() {
    println!("=== Multi-Objective Shapley Optimization + PATSAGi Simulation ===\n");

    let participants = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];

    let char_fn = |s: &HashSet<String>| -> f64 {
        let base = s.len() as f64 * 20.0;
        if s.contains("A") { base + 40.0 } else { base }
    };

    // Demonstrate multi-objective optimization
    let game = CooperativeGame::new(participants.clone(), char_fn);
    let (best_coalition, score) = game.optimize_coalition_multi_objective(3, 0.6, 0.4, 6);
    println!("Optimized coalition (fairness + value): {:?} (score: {:.2})", best_coalition, score);

    // Demonstrate integration with PATSAGi
    let mut mesh = lattice_conductor_v14::DistributedMercyMesh::new();
    let (decision, insight) = LatticeConductorEnhancements::submit_to_patsagi_with_game_theory(
        &mut mesh,
        "Strategic coalition decision",
        "Using multi-objective Shapley optimization",
        participants,
        char_fn,
    );

    println!("\nPATSAGi Decision: {:?}", decision);
    println!("Insight: {}", insight);

    println!("\n=== Simulation Complete ===");
}
