// examples/cooperative_governance_simulation.rs
// Comprehensive Simulation Suite for Cooperative Governance (v14.1+)
//
// Run with: cargo run --example cooperative_governance_simulation
//
// Focus: Simulate Shapley Value and Banzhaf Index across Ra-Thor relevant scenarios
// before deeper integration or formal documentation.

use lattice_conductor_v14::CooperativeGame;
use std::collections::HashSet;

fn main() {
    println!("=== Ra-Thor Cooperative Governance Simulation ===\n");

    // Scenario 1: Simple 3-player game (PATSAGi-style councils)
    println!("--- Scenario 1: 3-Player PATSAGi-style Council ---");
    let players1 = vec!["CouncilA".into(), "CouncilB".into(), "CouncilC".into()];
    let game1 = CooperativeGame::new(players1.clone(), |coalition| {
        match coalition.len() {
            3 => 100.0,           // Full council alignment
            2 => 60.0,            // Partial alignment
            _ => 0.0,
        }
    });

    let shapley1 = game1.shapley_value();
    let banzhaf1 = game1.banzhaf_index();
    print_results("3-Player Council", &shapley1, &banzhaf1);

    // Scenario 2: ONE Organism + Support Nodes (mercy-weighted)
    println!("\n--- Scenario 2: ONE Organism + Support Nodes ---");
    let players2 = vec!["UnifiedCore".into(), "Support1".into(), "Support2".into()];
    let game2 = CooperativeGame::new(players2.clone(), |coalition| {
        if coalition.contains("UnifiedCore") {
            80.0 + (coalition.len() as f64 - 1.0) * 10.0
        } else {
            coalition.len() as f64 * 15.0
        }
    });

    let shapley2 = game2.shapley_value();
    let banzhaf2 = game2.banzhaf_index();
    print_results("ONE Organism + Supports", &shapley2, &banzhaf2);

    // Scenario 3: Uneven contribution (tests fairness)
    println!("\n--- Scenario 3: Uneven Contribution Game ---");
    let players3 = vec!["HighContributor".into(), "Medium".into(), "Low".into()];
    let game3 = CooperativeGame::new(players3.clone(), |coalition| {
        let mut value = 0.0;
        if coalition.contains("HighContributor") { value += 70.0; }
        if coalition.contains("Medium") { value += 40.0; }
        if coalition.contains("Low") { value += 10.0; }
        value
    });

    let shapley3 = game3.shapley_value();
    let banzhaf3 = game3.banzhaf_index();
    print_results("Uneven Contribution", &shapley3, &banzhaf3);

    println!("\n=== Simulation Complete ===");
    println!("Use these results to validate behavior before deeper integration.");
}

fn print_results(name: &str, shapley: &[(String, f64)], banzhaf: &[(String, f64)]) {
    println!("\n[{}]", name);
    println!("Shapley Values:");
    for (player, value) in shapley {
        println!("  {}: {:.2}", player, value);
    }
    println!("Banzhaf Index (normalized):");
    for (player, value) in banzhaf {
        println!("  {}: {:.3}", player, value);
    }
}