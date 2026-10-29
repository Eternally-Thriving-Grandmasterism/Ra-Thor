// examples/cooperative_governance_simulation.rs
// Demonstrates structured GovernanceRiskReport logging

use lattice_conductor_v14::LatticeConductorEnhancements;
use std::collections::HashSet;

fn main() {
    println!("=== Structured Governance Risk Logging ===\n");

    let participants = vec!["Dominant".to_string(), "Weak1".to_string(), "Weak2".to_string()];
    let char_fn = |s: &HashSet<String>| -> f64 {
        if s.contains("Dominant") { 100.0 } else { 10.0 }
    };

    let mut mesh = lattice_conductor_v14::DistributedMercyMesh::new();

    let (decision, report) = LatticeConductorEnhancements::submit_to_patsagi_with_game_theory(
        &mut mesh,
        "Structured risk logging test",
        "Demonstrating GovernanceRiskReport",
        participants,
        char_fn,
    );

    println!("Final Decision: {:?}", decision);
    report.log();   // Structured logging

    println!("\n=== Simulation Complete ===");
}
