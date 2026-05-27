// examples/council_conflict_and_debate.rs
// Council Debate with Rebuttal Rounds
//
// Implements explicit debate phases + rebuttal rounds
// following PATSAGi principles (Mercy, Truth, ONE Organism, Evolution preference)

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision,
};
use std::collections::HashSet;

fn main() {
    println!("=== PATSAGi Council Debate with Rebuttal Rounds ===\n");

    // High-risk scenario
    let participants = vec!["Dominant".to_string(), "Weak1".to_string(), "Weak2".to_string()];
    let char_fn = |s: &HashSet<String>| -> f64 {
        if s.contains("Dominant") { 100.0 } else { 10.0 }
    };

    let game = CooperativeGame::new(participants.clone(), char_fn);
    let shapley = game.shapley_value();
    let banzhaf = game.banzhaf_index();

    let max_banzhaf = banzhaf.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
    let shapley_var = calculate_variance(&shapley);

    let risk_score = LatticeConductorEnhancements::calculate_governance_risk_score(
        max_banzhaf, shapley_var, 0.88
    );

    println!("Risk Score: {:.3} | Max Banzhaf: {:.3}", risk_score, max_banzhaf);

    if risk_score <= 0.55 {
        println!("Risk acceptable. No debate needed.");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "Full debate with rebuttals".to_string(),
    };

    // === Opening Statements ===
    println!("\n════════════════════════════════════════════════════════════");
    println!("                      OPENING STATEMENTS");
    println!("════════════════════════════════════════════════════════════");

    let mut positions: Vec<(&str, PatsagiDecision)> = vec![
        ("Mercy Council",   debate_mercy(&report)),
        ("Truth Council",   debate_truth(&report)),
        ("Justice Council", debate_justice(&report)),
        ("Harmony Council", debate_harmony(&report)),
        ("Council #13",     debate_council_13(&report)),
    ];

    for (name, decision) in &positions {
        println!("[{}] initial position: {:?}", name, decision);
    }

    // === Rebuttal Round 1 ===
    println!("\n════════════════════════════════════════════════════════════");
    println!("                      REBUTTAL ROUND 1");
    println!("════════════════════════════════════════════════════════════");

    // Simple rebuttal logic: Councils may shift based on others' positions
    for (name, decision) in positions.iter_mut() {
        match *name {
            "Mercy Council" => {
                // Mercy listens to Truth and may strengthen evolution call
                if matches!(decision, PatsagiDecision::Approved { .. }) {
                    if positions.iter().any(|(_, d)| matches!(d, PatsagiDecision::RequiresSelfEvolution { .. })) {
                        *decision = PatsagiDecision::RequiresSelfEvolution { priority: 2 };
                        println!("[Mercy Council] shifts after hearing Truth: Supports measured evolution.");
                    }
                }
            }
            "Justice Council" => {
                // Justice may call for arbitration if Council #13 is strict
                if matches!(positions.last().unwrap().1, PatsagiDecision::RequiresSelfEvolution { priority: 4 }) {
                    *decision = PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] };
                    println!("[Justice Council] aligns with Council #13 for stronger structural response.");
                }
            }
            _ => {}
        }
    }

    for (name, decision) in &positions {
        println!("[{}] after rebuttal: {:?}", name, decision);
    }

    // === Final Resolution (Weighted + Council #13 priority) ===
    println!("\n════════════════════════════════════════════════════════════");
    println!("                      FINAL RESOLUTION");
    println!("════════════════════════════════════════════════════════════");

    let final_decision = resolve_conflict_weighted(&positions, &report);
    println!("\nFinal Decision after debate and rebuttals: {:?}", final_decision);

    println!("\n=== Simulation Complete ===");
}

// Debate position functions

fn debate_mercy(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.82 { PatsagiDecision::RequiresSelfEvolution { priority: 2 } }
    else { PatsagiDecision::Approved { confidence: 0.82 } }
}

fn debate_truth(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.max_banzhaf > 0.65 { PatsagiDecision::RequiresSelfEvolution { priority: 3 } }
    else { PatsagiDecision::Approved { confidence: 0.88 } }
}

fn debate_justice(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.78 || report.max_banzhaf > 0.62 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] }
    } else {
        PatsagiDecision::Approved { confidence: 0.85 }
    }
}

fn debate_harmony(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.80 { PatsagiDecision::RequiresSelfEvolution { priority: 2 } }
    else { PatsagiDecision::Approved { confidence: 0.87 } }
}

fn debate_council_13(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.max_banzhaf > 0.60 || report.risk_score > 0.70 {
        PatsagiDecision::RequiresSelfEvolution { priority: 4 }
    } else {
        PatsagiDecision::Approved { confidence: 0.94 }
    }
}

// Weighted resolution with Council #13 priority and Mercy preference
fn resolve_conflict_weighted(
    positions: &[(&str, PatsagiDecision)],
    _report: &GovernanceRiskReport,
) -> PatsagiDecision {
    let c13 = positions.iter().find(|(n, _)| *n == "Council #13").unwrap();

    if matches!(c13.1, PatsagiDecision::RequiresCouncilArbitration { .. }) {
        return c13.1.clone();
    }

    let mut evolution_weight = 0;
    let mut arbitration_weight = 0;

    for (name, decision) in positions {
        let weight = if *name == "Council #13" { 3 } else { 1 };
        match decision {
            PatsagiDecision::RequiresSelfEvolution { .. } => evolution_weight += weight,
            PatsagiDecision::RequiresCouncilArbitration { .. } => arbitration_weight += weight,
            _ => {}
        }
    }

    if evolution_weight >= arbitration_weight {
        PatsagiDecision::RequiresSelfEvolution { priority: 3 }
    } else if arbitration_weight > 0 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] }
    } else {
        PatsagiDecision::Approved { confidence: 0.85 }
    }
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}
