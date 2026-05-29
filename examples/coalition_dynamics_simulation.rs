// examples/coalition_dynamics_simulation.rs
// Enhanced Coalition Dynamics Simulation
// Features: Sophisticated membership rules, GovernanceRiskReport logging, risk visualization, rapid concentration + self-correction behavior

use lattice_conductor_v14::{CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport};
use std::collections::HashSet;

fn main() {
    println!("=== Enhanced Coalition Dynamics Simulation ===\n");
    println!("Scenario: Rapid Power Concentration + Self-Correction\n");

    let mut current_players: Vec<String> = vec!["A".to_string(), "B".to_string()];
    let mut risk_history: Vec<f64> = Vec::new();

    for step in 1..=8 {
        println!("--- Step {} ---", step);
        println!("Coalition: {:?}", current_players);

        // Sophisticated characteristic function
        let char_fn = |s: &HashSet<String>| -> f64 {
            let mut value = s.len() as f64 * 12.0;
            if s.contains("A") { value += 30.0; }
            if s.contains("Dominant") { value += 80.0; }
            value
        };

        let game = CooperativeGame::new(current_players.clone(), char_fn);
        let shapley = game.shapley_value();
        let banzhaf = game.banzhaf_index();

        let max_banzhaf = banzhaf.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
        let shapley_var = calculate_variance(&shapley);

        // Use actual GovernanceRiskReport
        let risk_score = LatticeConductorEnhancements::calculate_governance_risk_score(
            max_banzhaf, shapley_var, 0.90
        );

        let report = GovernanceRiskReport {
            risk_score,
            max_banzhaf,
            shapley_variance: shapley_var,
            mercy_alignment: 0.90,
            recommended_action: if risk_score > 0.75 {
                "Escalate to Council #13".to_string()
            } else if risk_score > 0.55 {
                "Trigger self-evolution".to_string()
            } else {
                "No escalation".to_string()
            },
        };

        report.log();
        risk_history.push(risk_score);

        // === Sophisticated Membership Rules ===
        match step {
            2 => current_players.push("C".to_string()),           // Healthy growth
            3 => current_players.push("D".to_string()),
            5 => {                                                 // Rapid power concentration begins
                current_players = vec![
                    "Dominant".to_string(),
                    "Weak1".to_string(),
                    "Weak2".to_string(),
                ];
            }
            6 => current_players.push("Weak3".to_string()),      // Further concentration
            7 => {                                                 // Self-correction trigger
                if risk_score > 0.70 {
                    println!("[SELF-CORRECTION] High risk detected. Coalition rebalancing...");
                    current_players = vec!["A".to_string(), "B".to_string(), "C".to_string()];
                }
            }
            _ => {}
        }

        println!();
    }

    // Simple text visualization of risk score over time
    println!("=== Risk Score Over Time ===");
    for (i, score) in risk_history.iter().enumerate() {
        let bar = "█".repeat((score * 20.0) as usize);
        println!("Step {:>2}: {:.3} {}", i + 1, score, bar);
    }

    println!("\n=== Simulation Complete ===");
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}
