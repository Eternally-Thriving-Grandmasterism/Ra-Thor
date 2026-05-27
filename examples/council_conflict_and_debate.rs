// examples/council_conflict_and_debate.rs
// Light exploration of Council Conflict Resolution + Multi-Agent Debate
//
// Multiple PATSAGi archetypes review the same situation, produce conflicting decisions,
// and apply simple resolution mechanisms.

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision,
};
use std::collections::HashSet;

fn main() {
    println!("=== Council Conflict + Light Debate Simulation ===\n");

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
        println!("Risk acceptable. No action needed.");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "Resolve council conflict on self-correction".to_string(),
    };

    let review_request = PatsagiReviewRequest {
        topic: "Conflicting council views on self-correction".to_string(),
        summary: "Multiple councils disagree on appropriate response to power concentration.".to_string(),
        mercy_impact_score: report.mercy_alignment,
        requested_by: "lattice-conductor".to_string(),
    };

    println!("\n--- Individual Council Positions ---");

    let mercy   = review_mercy(&review_request, &report);
    let truth   = review_truth(&review_request, &report);
    let justice = review_justice(&review_request, &report);
    let c13     = review_council_13(&review_request, &report);

    println!("Mercy:   {:?}", mercy);
    println!("Truth:   {:?}", truth);
    println!("Justice: {:?}", justice);
    println!("C#13:    {:?}", c13);

    // === Simple Conflict Resolution ===
    println!("\n--- Conflict Resolution ---");

    // Priority-based resolution (Council #13 has highest authority)
    let final_decision = if matches!(c13, PatsagiDecision::RequiresCouncilArbitration { .. }) {
        c13.clone()
    } else if matches!(justice, PatsagiDecision::RequiresCouncilArbitration { .. }) {
        justice.clone()
    } else if matches!(truth, PatsagiDecision::RequiresSelfEvolution { .. }) ||
              matches!(mercy, PatsagiDecision::RequiresSelfEvolution { .. }) {
        // Prefer stronger self-evolution signal
        if matches!(c13, PatsagiDecision::RequiresSelfEvolution { .. }) {
            c13.clone()
        } else {
            truth.clone()
        }
    } else {
        c13.clone()
    };

    println!("\nFinal Resolved Decision: {:?}", final_decision);

    println!("\n=== Simulation Complete ===");
}

// Archetype review functions (simplified for conflict demo)

fn review_mercy(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.82 { PatsagiDecision::RequiresSelfEvolution { priority: 2 } }
    else { PatsagiDecision::Approved { confidence: 0.82 } }
}

fn review_truth(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.max_banzhaf > 0.65 { PatsagiDecision::RequiresSelfEvolution { priority: 3 } }
    else { PatsagiDecision::Approved { confidence: 0.88 } }
}

fn review_justice(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.78 || report.max_banzhaf > 0.62 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] }
    } else {
        PatsagiDecision::Approved { confidence: 0.85 }
    }
}

fn review_council_13(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.max_banzhaf > 0.60 || report.risk_score > 0.70 {
        PatsagiDecision::RequiresSelfEvolution { priority: 4 }
    } else {
        PatsagiDecision::Approved { confidence: 0.94 }
    }
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}
