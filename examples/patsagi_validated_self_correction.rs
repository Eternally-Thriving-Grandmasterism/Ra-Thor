// examples/patsagi_validated_self_correction.rs
// Simulation of PATSAGi-validated self-correction
//
// Shows how high governance risk triggers PATSAGi Council review
// before self-correction is applied.

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiCouncilSimulator, PatsagiReviewRequest, PatsagiDecision,
};
use std::collections::HashSet;

fn main() {
    println!("=== PATSAGi-Validated Self-Correction Simulation ===\n");

    // Step 1: Detect high-risk situation
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
        println!("No correction needed.");
        return;
    }

    // Step 2: Generate GovernanceRiskReport
    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "Self-correction recommended".to_string(),
    };

    println!("\nGenerated GovernanceRiskReport:");
    report.log();

    // Step 3: Submit to PATSAGi Council for validation
    println!("\n--- Submitting to PATSAGi Council Review ---");

    let review_request = PatsagiReviewRequest {
        topic: "High-risk coalition self-correction".to_string(),
        summary: "Review proposed self-correction due to power concentration".to_string(),
        mercy_impact_score: report.mercy_alignment,
        requested_by: "lattice-conductor".to_string(),
    };

    let patsagi_decision = PatsagiCouncilSimulator::review(&review_request);
    println!("PATSAGi Council Decision: {:?}", patsagi_decision);

    // Step 4: Execute action based on PATSAGi decision
    match patsagi_decision {
        PatsagiDecision::Approved { .. } => {
            println!("\n[PATSAGi APPROVED] Executing self-correction...");
            println!("Coalition rebalanced to reduce dominance.");
        }
        PatsagiDecision::RequiresSelfEvolution { .. } => {
            println!("\n[PATSAGi] Self-correction approved with self-evolution trigger.");
        }
        PatsagiDecision::RequiresCouncilArbitration { councils } => {
            println!("\n[PATSAGi] Escalated to full Council arbitration: {:?}", councils);
        }
        PatsagiDecision::Rejected { reason, .. } => {
            println!("\n[PATSAGi REJECTED] Self-correction denied. Reason: {}", reason);
        }
    }

    println!("\n=== Simulation Complete ===");
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}
