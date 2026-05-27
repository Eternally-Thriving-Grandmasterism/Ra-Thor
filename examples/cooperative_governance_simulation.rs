// examples/cooperative_governance_simulation.rs
// Full simulation of multi-objective Shapley optimization + PATSAGi integration

use lattice_conductor_v14::{LatticeConductorEnhancements, CooperativeGame};
use std::collections::HashSet;

fn main() {
    println!("=== Multi-Objective Shapley + PATSAGi Simulation ===\n");

    let participants = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];

    let char_fn = |s: &HashSet<String>| -> f64 {
        s.len() as f64 * 25.0 + if s.contains("A") { 30.0 } else { 0.0 }
    };

    let game = CooperativeGame::new(participants.clone(), char_fn);

    // Multi-objective optimization
    let (best_coalition, score) = game.optimize_coalition_multi_objective(3, 0.6, 0.4, 8);
    println!("Best fair+value coalition: {:?} (score: {:.2})", best_coalition, score);

    // Integrate with PATSAGi flow
    let mut mesh = lattice_conductor_v14::DistributedMercyMesh::new();
    let (decision, insight) = LatticeConductorEnhancements::submit_to_patsagi_with_game_theory(
        &mut mesh,
        "Coalition optimization test",
        "Using multi-objective Shapley",
        participants,
        char_fn,
    );

    println!("PATSAGi Decision: {:?}", decision);
    println!("Insight: {}", insight);

    println!("\n=== Simulation Complete ===");
}
