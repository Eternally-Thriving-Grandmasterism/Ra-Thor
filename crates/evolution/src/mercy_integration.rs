/// Initial Mercy Orchestrator integration layer (Phase C).
///
/// This module begins the systemic expansion by providing a clean interface
/// to the Mercy Orchestrator for more sophisticated Mercy-aware decision making.
///
/// Future work will include deeper valence routing, gate escalation,
/// and PATSAGi Council integration.

use tracing::info;

/// High-level Mercy decision result.
#[derive(Debug, Clone)]
pub struct MercyDecision {
    pub is_acceptable: bool,
    pub sovereignty_score: f32,
    pub overall_valence: f32,
    pub reasoning: String,
}

/// Perform a Mercy Orchestrator-level decision on an already evaluated proposal.
/// This is the first step toward deeper Mercy integration.
pub fn mercy_decide(
    evaluation_result: &crate::evaluation::EvaluationResult,
) -> MercyDecision {
    info!(
        is_acceptable = evaluation_result.is_acceptable(),
        sovereignty = evaluation_result.sovereignty_gate_score,
        "Performing Mercy Orchestrator decision"
    );

    // Placeholder for future full Mercy Orchestrator integration
    // Currently uses the evaluator's is_acceptable() as base decision
    let is_acceptable = evaluation_result.is_acceptable();

    MercyDecision {
        is_acceptable,
        sovereignty_score: evaluation_result.sovereignty_gate_score,
        overall_valence: evaluation_result.average_mercy_score,
        reasoning: if is_acceptable {
            "Proposal meets current Mercy thresholds".to_string()
        } else {
            "Proposal failed one or more Mercy thresholds".to_string()
        },
    }
}

/// Check if a proposal should be escalated to PATSAGi Councils
/// for multi-perspective review (future enhancement).
pub fn should_escalate_to_patsagi(
    decision: &MercyDecision,
) -> bool {
    // Placeholder logic - future versions will use more sophisticated criteria
    !decision.is_acceptable && decision.sovereignty_score < 5.0
}
