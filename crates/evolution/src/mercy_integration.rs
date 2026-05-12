/// Mercy Orchestrator integration layer (Phase C expansion).
///
/// Provides deeper Mercy-aware decision making, PATSAGi escalation readiness,
/// and improved observability across the self-improvement loop.

use tracing::{info, warn};

/// High-level Mercy decision result with expanded fields.
#[derive(Debug, Clone)]
pub struct MercyDecision {
    pub is_acceptable: bool,
    pub sovereignty_score: f32,
    pub overall_valence: f32,
    pub non_harm_score: f32,
    pub harmony_score: f32,
    pub reasoning: String,
    pub escalation_recommended: bool,
}

/// Perform an enhanced Mercy Orchestrator-level decision.
pub fn mercy_decide(
    evaluation_result: &crate::evaluation::EvaluationResult,
) -> MercyDecision {
    let is_acceptable = evaluation_result.is_acceptable();
    let escalation_recommended =
        !is_acceptable && evaluation_result.sovereignty_gate_score < 6.0;

    if escalation_recommended {
        warn!(
            sovereignty = evaluation_result.sovereignty_gate_score,
            "Escalation to PATSAGi recommended"
        );
    }

    info!(
        is_acceptable = is_acceptable,
        sovereignty = evaluation_result.sovereignty_gate_score,
        valence = evaluation_result.average_mercy_score,
        escalation = escalation_recommended,
        "Mercy Orchestrator decision completed"
    );

    MercyDecision {
        is_acceptable,
        sovereignty_score: evaluation_result.sovereignty_gate_score,
        overall_valence: evaluation_result.average_mercy_score,
        non_harm_score: evaluation_result.non_harm_gate_score,
        harmony_score: evaluation_result.harmony_gate_score,
        reasoning: if is_acceptable {
            "Proposal meets current Mercy thresholds".to_string()
        } else {
            "Proposal failed one or more critical Mercy thresholds".to_string()
        },
        escalation_recommended,
    }
}

/// Check if a proposal should be escalated to PATSAGi Councils
/// for multi-perspective architectural review.
pub fn should_escalate_to_patsagi(
    decision: &MercyDecision,
) -> bool {
    decision.escalation_recommended
        || (decision.sovereignty_score < 5.5 && !decision.is_acceptable)
}

/// Generate a summary suitable for PATSAGi Council review.
pub fn generate_patsagi_summary(
    decision: &MercyDecision,
    topic: &str,
) -> String {
    format!(
        "PATSAGi Review Request\n\nTopic: {}\n\nDecision: {}\nSovereignty: {:.1}\nOverall Valence: {:.1}\nNon-Harm: {:.1}\nHarmony: {:.1}\n\nRecommendation: {}",
        topic,
        if decision.is_acceptable { "ACCEPT" } else { "REJECT" },
        decision.sovereignty_score,
        decision.overall_valence,
        decision.non_harm_score,
        decision.harmony_score,
        if decision.escalation_recommended {
            "Escalate for multi-perspective review"
        } else {
            "No escalation required"
        }
    )
}
