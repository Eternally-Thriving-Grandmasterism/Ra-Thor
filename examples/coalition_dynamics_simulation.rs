// examples/coalition_dynamics_simulation.rs
// Step-by-step simulation of coalition dynamics
//
// Shows how Shapley values, Banzhaf power, and Governance Risk Score evolve
// as a coalition changes over time.

use lattice_conductor_v14::{CooperativeGame, LatticeConductorEnhancements};
use std::collections::HashSet;

fn main() {
    println!("=== Coalition Dynamics Simulation ===\n");

    let mut current_players: Vec<String> = vec!["A".to_string(), "B".to_string()];

    for step in 1..=6 {
        println!("--- Step {} ---", step);
        println!("Current coalition: {:?}", current_players);

        let char_fn = |s: &HashSet<String>| -> f64 {
            let base = s.len() as f64 * 15.0;
            if s.contains("A") { base + 35.0 } else { base }
        };

        let game = CooperativeGame::new(current_players.clone(), char_fn);

        // Calculate metrics
        let shapley = game.shapley_value();
        let banzhaf = game.banzhaf_index();
        let max_banzhaf = banzhaf.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
        let shapley_var = calculate_shapley_variance(&shapley);

        // Composite risk score
        let risk_score = LatticeConductorEnhancements::calculate_governance_risk_score(
            max_banzhaf, shapley_var, 0.92
        );

        println!("Shapley: {:?}", shapley);
        println!("Max Banzhaf: {:.3}", max_banzhaf);
        println!("Risk Score: {:.3}", risk_score);

        // Simulate PATSAGi-style decision
        let decision = if risk_score > 0.75 {
            "RequiresCouncilArbitration"
        } else if risk_score > 0.55 {
            "RequiresSelfEvolution"
        } else {
            "Approved"
        };
        println!("PATSAGi Decision: {}\n", decision);

        // Dynamic membership change
        if step == 2 {
            current_players.push("C".to_string());
        }
        if step == 4 {
            current_players.push("D".to_string());
        }
        if step == 5 {
            // Simulate power concentration scenario
            current_players = vec!["Dominant".to_string(), "Weak1".to_string(), "Weak2".to_string()];
        }
    }

    println!("=== Simulation Complete ===");
}

fn calculate_shapley_variance(shapley: &[(String, f64)]) -> f64 {
    if shapley.is_empty() { return 0.0; }
    let mean = shapley.iter().map(|(_, v)| *v).sum::<f64>() / shapley.len() as f64;
    shapley.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / shapley.len() as f64
}
