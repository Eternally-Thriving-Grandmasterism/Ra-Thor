// examples/patsagi_validated_self_correction.rs
// PATSAGi-Validated Self-Correction with Expanded Council Archetypes
//
// Features: Mercy, Truth, Justice, Harmony, and Council #13 archetypes
// with structured, readable output and verbose reasoning.

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision,
};
use std::collections::HashSet;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     PATSAGi-Validated Self-Correction Simulation           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

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

    println!("Risk Score: {:.3}   |   Max Banzhaf: {:.3}", risk_score, max_banzhaf);

    if risk_score <= 0.55 {
        println!("Risk is within acceptable limits. No correction required.");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "PATSAGi-validated self-correction".to_string(),
    };

    println!("\n--- Governance Risk Report ---");
    report.log();

    let review_request = PatsagiReviewRequest {
        topic: "High-risk coalition self-correction".to_string(),
        summary: "Evaluate proposed self-correction due to detected power concentration.".to_string(),
        mercy_impact_score: report.mercy_alignment,
        requested_by: "lattice-conductor".to_string(),
    };

    println!("\n════════════════════════════════════════════════════════════");
    println!("                    COUNCIL ARCHETYPE REVIEWS");
    println!("════════════════════════════════════════════════════════════");

    let mercy   = review_mercy_council(&review_request, &report);
    let truth   = review_truth_council(&review_request, &report);
    let justice = review_justice_council(&review_request, &report);
    let harmony = review_harmony_council(&review_request, &report);
    let c13     = review_council_13(&review_request, &report);

    println!("\n[ Mercy Council ]     {:?}", mercy);
    println!("[ Truth Council ]     {:?}", truth);
    println!("[ Justice Council ]   {:?}", justice);
    println!("[ Harmony Council ]   {:?}", harmony);
    println!("[ Council #13 ]       {:?}", c13);

    println!("\n════════════════════════════════════════════════════════════");
    println!("                    FINAL RULING (Council #13)");
    println!("════════════════════════════════════════════════════════════");

    match c13 {
        PatsagiDecision::Approved { confidence } => {
            println!("Self-correction APPROVED (confidence: {:.2})", confidence);
        }
        PatsagiDecision::RequiresSelfEvolution { priority } => {
            println!("Self-correction + Self-Evolution triggered (priority: {})", priority);
        }
        PatsagiDecision::RequiresCouncilArbitration { councils } => {
            println!("Escalated to full PATSAGi arbitration: {:?}", councils);
        }
        PatsagiDecision::Rejected { reason, .. } => {
            println!("Self-correction REJECTED. Reason: {}", reason);
        }
    }

    println!("\n════════════════════════════════════════════════════════════");
    println!("                         SIMULATION END");
    println!("════════════════════════════════════════════════════════════");
}

// ==================== Council Archetypes ====================

fn review_mercy_council(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.82 {
        PatsagiDecision::RequiresSelfEvolution { priority: 2 }
    } else {
        PatsagiDecision::Approved { confidence: 0.82 }
    }
}

fn review_truth_council(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.max_banzhaf > 0.65 {
        PatsagiDecision::RequiresSelfEvolution { priority: 3 }
    } else {
        PatsagiDecision::Approved { confidence: 0.88 }
    }
}

fn review_justice_council(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.78 || report.max_banzhaf > 0.62 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] }
    } else {
        PatsagiDecision::Approved { confidence: 0.85 }
    }
}

fn review_harmony_council(_req: &PatsagiReviewRequest, report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.80 {
        PatsagiDecision::RequiresSelfEvolution { priority: 2 }
    } else {
        PatsagiDecision::Approved { confidence: 0.87 }
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
