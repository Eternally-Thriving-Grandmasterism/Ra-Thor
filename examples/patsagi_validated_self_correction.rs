// examples/patsagi_validated_self_correction.rs
// PATSAGi-Validated Self-Correction with Council Archetypes
//
// Demonstrates how different PATSAGi Councils (including Council #13)
// can review and decide on self-correction with varying reasoning.

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision,
};
use std::collections::HashSet;

fn main() {
    println!("=== PATSAGi-Validated Self-Correction Simulation ===\n");

    // === High-Risk Situation ===
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

    println!("Risk Score Detected: {:.3}", risk_score);

    if risk_score <= 0.55 {
        println!("Risk is acceptable. No correction needed.");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "Self-correction with PATSAGi validation".to_string(),
    };

    println!("\nGenerated Risk Report:");
    report.log();

    // === Create Review Request ===
    let review_request = PatsagiReviewRequest {
        topic: "High-risk coalition self-correction".to_string(),
        summary: "Power concentration detected. Review proposed self-correction.".to_string(),
        mercy_impact_score: report.mercy_alignment,
        requested_by: "lattice-conductor".to_string(),
    };

    // === Council Archetype Reviews ===
    println!("\n=== Council Reviews ===");

    // Regular Council Review
    let regular_decision = simulate_regular_council_review(&review_request, &report);
    println!("\n[Regular Council] Decision: {:?}", regular_decision);

    // Council #13 (Supreme Architect) Review
    let council_13_decision = simulate_council_13_review(&review_request, &report);
    println!("\n[Council #13 - Supreme Architect] Decision: {:?}", council_13_decision);

    // Final outcome based on Council #13 (highest authority in this simulation)
    println!("\n=== Final Outcome (Council #13 ruling) ===");
    match council_13_decision {
        PatsagiDecision::Approved { confidence } => {
            println!("Self-correction APPROVED (confidence: {:.2}). Executing rebalancing...", confidence);
        }
        PatsagiDecision::RequiresSelfEvolution { priority } => {
            println!("Self-correction approved with Self-Evolution trigger (priority {}).", priority);
        }
        PatsagiDecision::RequiresCouncilArbitration { councils } => {
            println!("Escalated to full arbitration: {:?}", councils);
        }
        PatsagiDecision::Rejected { reason, .. } => {
            println!("Self-correction REJECTED. Reason: {}", reason);
        }
    }

    println!("\n=== Simulation Complete ===");
}

/// Regular Council Review (more balanced / pragmatic)
fn simulate_regular_council_review(
    request: &PatsagiReviewRequest,
    report: &GovernanceRiskReport,
) -> PatsagiDecision {
    if report.risk_score > 0.80 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] }
    } else if report.max_banzhaf > 0.65 {
        PatsagiDecision::RequiresSelfEvolution { priority: 1 }
    } else {
        PatsagiDecision::Approved { confidence: 0.85 }
    }
}

/// Council #13 Review (Supreme Architect - stricter, long-term focused)
fn simulate_council_13_review(
    request: &PatsagiReviewRequest,
    report: &GovernanceRiskReport,
) -> PatsagiDecision {
    if report.risk_score > 0.70 || report.max_banzhaf > 0.60 {
        // Council #13 is stricter on power concentration
        PatsagiDecision::RequiresSelfEvolution { priority: 3 }
    } else if report.risk_score > 0.85 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![13] }
    } else {
        PatsagiDecision::Approved { confidence: 0.92 }
    }
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}
