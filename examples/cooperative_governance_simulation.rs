// examples/cooperative_governance_simulation.rs
// Simulation covering refined PATSAGi + Game Theory integration + Thunder Lattice extension

use lattice_conductor_v14::{
    LatticeConductorEnhancements,
    DistributedMercyMesh,
};
use std::collections::HashSet;

fn main() {
    println!("=== Refined PATSAGi + Cooperative Game Theory Simulation ===\n");

    let mut mesh = DistributedMercyMesh::new();

    let participants = vec![
        "CouncilA".to_string(),
        "CouncilB".to_string(),
        "DominantNode".to_string(),
    ];

    // Characteristic function that gives high value to coalitions including DominantNode
    let char_fn = |coalition: &HashSet<String>| -> f64 {
        if coalition.contains("DominantNode") {
            90.0 + coalition.len() as f64 * 5.0
        } else {
            coalition.len() as f64 * 20.0
        }
    };

    let (decision, insight) = LatticeConductorEnhancements::submit_to_patsagi_with_game_theory(
        &mut mesh,
        "High-stakes coalition decision",
        "Test power concentration and contribution fairness",
        participants,
        char_fn,
    );

    println!("Final Influenced Decision: {:?}", decision);
    println!("Game Theory Insight: {}", insight);

    println!("\n=== Simulation Complete ===");
}
