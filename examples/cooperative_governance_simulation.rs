// examples/cooperative_governance_simulation.rs
// Simulation with prominent Risk Score logging

use lattice_conductor_v14::{LatticeConductorEnhancements, CooperativeGame};
use std::collections::HashSet;

fn main() {
    println!("=== Governance Risk Score Simulation ===\n");

    // Scenario: Power concentration test
    let participants = vec!["Dominant".to_string(), "Weak1".to_string(), "Weak2".to_string()];
    let char_fn = |s: &HashSet<String>| -> f64 {
        if s.contains("Dominant") { 100.0 } else { 10.0 }
    };

    let mut mesh = lattice_conductor_v14::DistributedMercyMesh::new();

    let (decision, insight) = LatticeConductorEnhancements::submit_to_patsagi_with_game_theory(
        &mut mesh,
        "Power concentration analysis",
        "Testing composite risk score visibility",
        participants,
        char_fn,
    );

    println!("Final Decision: {:?}", decision);
    println!("Risk Score Insight: {}", insight);

    println!("\n=== Simulation Complete ===");
}
