// examples/council_conflict_and_debate.rs
// Phase 2: Argument Credibility Scoring + Dynamic Persuasion

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision, LogicalFallacyDetector, ArgumentGraph,
};
use std::collections::HashSet;

/// Calculate Argument Credibility Score (0.0 - 1.0)
fn calculate_argument_credibility(
    effective_strength: f64,
    conflict_level: f64,
    fallacy_penalty: f64,
) -> f64 {
    let base = effective_strength * 0.6;
    let conflict_adjustment = (1.0 - conflict_level) * 0.3;
    let credibility = (base + conflict_adjustment) * (1.0 - fallacy_penalty);
    credibility.clamp(0.0, 1.0)
}

fn main() {
    println!("=== Phase 2: Argument Credibility Scoring ===\n");
    println!("Mates! Persuasion now uses a full credibility score.\n");

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
        println!("Risk acceptable.\n");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "Argument Credibility Scoring".to_string(),
    };

    let mut arg_graph = ArgumentGraph::new();
    let main_claim = arg_graph.add_claim(
        "Strong self-evolution required due to power concentration".to_string(),
        "Council #13".to_string(),
        0.85,
    );

    arg_graph.add_support(main_claim, main_claim, "Supports coherence".to_string(), "Truth Council".to_string(), 0.8);
    arg_graph.add_attack(main_claim, main_claim, "Risk of disruption".to_string(), "Justice Council".to_string(), 0.3);

    let effective = arg_graph.effective_strength(main_claim).unwrap_or(0.5);
    let conflict = arg_graph.conflict_level(main_claim).unwrap_or(0.0);

    // Simulate previous fallacy detection
    let fallacy_penalty = 0.15; // Example: fallacies were detected earlier

    let credibility = calculate_argument_credibility(effective, conflict, fallacy_penalty);
    println!("\n[Credibility Score] {:.2} (Strength: {:.2}, Conflict: {:.2}, Fallacy Penalty: {:.2})", 
             credibility, effective, conflict, fallacy_penalty);

    // ROUND 1
    println!("\n--- ROUND 1: Opening Statements ---");
    let mut positions: Vec<(&str, PatsagiDecision)> = vec![
        ("Mercy Council",   debate_mercy(&report)),
        ("Truth Council",   debate_truth(&report)),
        ("Justice Council", debate_justice(&report)),
        ("Council #13",     debate_council_13(&report)),
    ];

    for (name, decision) in &positions {
        println!("[{}] : {:?}", name, decision);
    }

    // ROUND 2 - Uses Credibility Score
    println!("\n--- ROUND 2: Credibility-Based Persuasion ---");

    let c13_pos = positions.iter().find(|(n, _)| *n == "Council #13").unwrap().1.clone();

    for (name, decision) in positions.iter_mut() {
        if *name == "Council #13" { continue; }

        let base_sensitivity = match *name {
            "Mercy Council" | "Harmony Council" => 0.8,
            "Truth Council" => 0.7,
            "Justice Council" => 0.5,
            _ => 0.6,
        };

        // Dynamic weight influenced by credibility
        let dynamic_weight = base_sensitivity * credibility;

        if dynamic_weight > 0.5 {
            if matches!(c13_pos, PatsagiDecision::RequiresSelfEvolution { .. }) {
                if matches!(decision, PatsagiDecision::Approved { .. }) {
                    *decision = PatsagiDecision::RequiresSelfEvolution { priority: 2 };
                    println!("[{}] persuaded by credible argument (weight: {:.2}).", name, dynamic_weight);
                }
            }
        }
    }

    for (name, decision) in &positions {
        println!("[{}] after Round 2: {:?}", name, decision);
    }

    println!("\n--- FINAL RESOLUTION ---");
    let final = resolve_conflict_weighted(&positions, &report);
    println!("Final Decision: {:?}", final);
    println!("\nWe move forward with credibility-weighted evolution, Mates!\n");

    println!("=== Phase 2 Progress ===");
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
