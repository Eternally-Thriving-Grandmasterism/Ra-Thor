// examples/patsagi_validated_self_correction.rs
// PATSAGi-Validated Self-Correction with Multiple Council Archetypes
//
// Demonstrates how different PATSAGi Council archetypes review and decide
// on self-correction with detailed reasoning.

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision,
};
use std::collections::HashSet;

fn main() {
    println!("=== PATSAGi-Validated Self-Correction with Council Archetypes ===\n");

    // High-risk situation
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
        println!("Risk acceptable. No correction needed.");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "PATSAGi-validated self-correction".to_string(),
    };

    println!("\nRisk Report:");
    report.log();

    let review_request = PatsagiReviewRequest {
        topic: "High-risk self-correction review".to_string(),
        summary: "Power concentration detected. Evaluate self-correction proposal.".to_string(),
        mercy_impact_score: report.mercy_alignment,
        requested_by: "lattice-conductor".to_string(),
    };

    println!("\n=== Council Archetype Reviews ===");

    // Review by multiple archetypes
    let mercy_council = review_by_mercy_council(&review_request, &report);
    let truth_council = review_by_truth_council(&review_request, &report);
    let council_13 = review_by_council_13(&review_request, &report);

    println!("\n[Mercy Council]      {:?}", mercy_council);
    println!("[Truth Council]     {:?}", truth_council);
    println!("[Council #13]       {:?}", council_13);

    // Final decision (Council #13 as highest authority)
    println!("\n=== Final Ruling (Council #13) ===");
    match council_13 {
        PatsagiDecision::Approved { confidence } => {
            println!("Self-correction APPROVED (confidence {:.2})", confidence);
        }
        PatsagiDecision::RequiresSelfEvolution { priority } => {
            println!("Requires Self-Evolution (priority {})", priority);
        }
        PatsagiDecision::RequiresCouncilArbitration { councils } => {
            println!("Escalated to full arbitration: {:?}", councils);
        }
        PatsagiDecision::Rejected { reason, .. } => {
            println!("REJECTED. Reason: {}", reason);
        }
    }

    println!("\n=== Simulation Complete ===");
}

// === Council Archetypes with Verbose Reasoning ===

fn review_by_mercy_council(request: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    println!("\n[Mercy Council Reasoning]");
    println!("  Focus: Compassion, harm reduction, and redemption.");
    if report.risk_score > 0.85 {
        println!("  Decision: High risk requires strong intervention for protection.");
        PatsagiDecision::RequiresSelfEvolution { priority: 2 }
    } else {
        println!("  Decision: Mercy supports giving the coalition a chance to self-correct.");
        PatsagiDecision::Approved { confidence: 0.80 }
    }
}

fn review_by_truth_council(request: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    println!("\n[Truth Council Reasoning]");
    println!("  Focus: Structural integrity, long-term truth, and systemic health.");
    if report.max_banzhaf > 0.65 {
        println!("  Decision: Structural imbalance is too high. Self-evolution strongly recommended.");
        PatsagiDecision::RequiresSelfEvolution { priority: 3 }
    } else {
        PatsagiDecision::Approved { confidence: 0.88 }
    }
}

fn review_by_council_13(request: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    println!("\n[Council #13 - Supreme Architect Reasoning]");
    println!("  Focus: ONE Organism coherence, long-term stability, and highest standards.");
    if report.max_banzhaf > 0.60 || report.risk_score > 0.70 {
        println!("  Decision: Power concentration threatens ONE Organism integrity. Self-evolution required.");
        PatsagiDecision::RequiresSelfEvolution { priority: 4 }
    } else {
        PatsagiDecision::Approved { confidence: 0.93 }
    }
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}
