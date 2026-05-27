// examples/council_conflict_and_debate.rs
// Persuasion Dynamics in PATSAGi Council Debate
//
// Councils can influence each other during debate rounds.
// Council #13 has highest persuasion power. Archetype alignment affects influence.

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision,
};
use std::collections::HashSet;

fn main() {
    println!("=== Persuasion Dynamics in PATSAGi Council Debate ===\n");

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
        recommended_action: "Debate with persuasion dynamics".to_string(),
    };

    let review_request = PatsagiReviewRequest {
        topic: "Persuasion dynamics test".to_string(),
        summary: "Observe how councils influence each other.".to_string(),
        mercy_impact_score: report.mercy_alignment,
        requested_by: "lattice-conductor".to_string(),
    };

    // Initial positions
    let mut positions: Vec<(&str, PatsagiDecision)> = vec![
        ("Mercy Council",   debate_mercy(&report)),
        ("Truth Council",   debate_truth(&report)),
        ("Justice Council", debate_justice(&report)),
        ("Harmony Council", debate_harmony(&report)),
        ("Council #13",     debate_council_13(&report)),
    ];

    println!("\n--- Opening Positions ---");
    for (name, decision) in &positions {
        println!("{}: {:?}", name, decision);
    }

    // === Persuasion Round ===
    println!("\n--- Persuasion Round ---");

    // Council #13 has high persuasion power
    let c13_decision = positions.iter().find(|(n, _)| *n == "Council #13").unwrap().1.clone();

    for (name, decision) in positions.iter_mut() {
        if *name == "Council #13" { continue; }

        let persuasion_strength = match *name {
            "Mercy Council" | "Harmony Council" => 0.6, // More open to influence
            "Truth Council" | "Justice Council" => 0.4,
            _ => 0.3,
        };

        // Strong influence from Council #13
        if matches!(c13_decision, PatsagiDecision::RequiresSelfEvolution { priority: 4 }) {
            if persuasion_strength > 0.5 {
                *decision = PatsagiDecision::RequiresSelfEvolution { priority: 3 };
                println!("[{}] persuaded by Council #13 toward stronger evolution.", name);
            }
        }
    }

    println!("\n--- Positions After Persuasion ---");
    for (name, decision) in &positions {
        println!("{}: {:?}", name, decision);
    }

    // Final resolution
    let final = resolve_conflict_weighted(&positions, &report);
    println!("\nFinal Decision: {:?}", final);

    println!("\n=== Simulation Complete ===");
}

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

fn resolve_conflict_weighted(positions: &[(&str, PatsagiDecision)], _report: &GovernanceRiskReport) -> PatsagiDecision {
    let c13 = positions.iter().find(|(n, _)| *n == "Council #13").unwrap();
    if matches!(c13.1, PatsagiDecision::RequiresCouncilArbitration { .. }) {
        return c13.1.clone();
    }

    let mut evolution = 0;
    let mut arbitration = 0;

    for (name, d) in positions {
        let w = if *name == "Council #13" { 3 } else { 1 };
        match d {
            PatsagiDecision::RequiresSelfEvolution { .. } => evolution += w,
            PatsagiDecision::RequiresCouncilArbitration { .. } => arbitration += w,
            _ => {}
        }
    }

    if evolution >= arbitration {
        PatsagiDecision::RequiresSelfEvolution { priority: 3 }
    } else if arbitration > 0 {
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
