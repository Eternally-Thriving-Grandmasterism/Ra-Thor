/// Production-grade integration layer for `self-improvement-extensions`.
///
/// Provides clean, observable, and Mercy-aligned access to the refined
/// TOLC + 7 Living Mercy Gates evaluator from within the evolution system.
///
/// This module is designed for seamless use in self-improvement loops.

use self_improvement_extensions::{evaluate_proposal_with_tolc_and_mercy, EvaluationResult};
use tracing::{info, instrument};

/// Re-export core types for convenience
pub use self_improvement_extensions::EvaluationResult;

/// Evaluate a proposal using the full TOLC + Mercy pipeline
/// and return a structured result with rich observability.
///
/// This is the primary entry point for evaluation inside the evolution system.
#[instrument(skip(model), fields(proposal_len = proposal.len()))]
pub fn evaluate_proposal(
    model: &llama_cpp_gguf::LlamaModel,
    proposal: &str,
) -> EvaluationResult {
    info!("Evaluating proposal through TOLC + Mercy pipeline");

    let result = evaluate_proposal_with_tolc_and_mercy(model, proposal);

    info!(
        average_tolc = result.average_tolc_score,
        average_mercy = result.average_mercy_score,
        sovereignty = result.sovereignty_gate_score,
        passes_threshold = result.passes_threshold,
        is_acceptable = result.is_acceptable(),
        "Proposal evaluation completed"
    );

    result
}

/// Convenience helper: Evaluate and immediately check if the proposal is acceptable.
///
/// Returns the full `EvaluationResult` along with a boolean for quick branching.
pub fn evaluate_and_decide(
    model: &llama_cpp_gguf::LlamaModel,
    proposal: &str,
) -> (EvaluationResult, bool) {
    let result = evaluate_proposal(model, proposal);
    let acceptable = result.is_acceptable();

    if acceptable {
        info!("Proposal accepted by Mercy and TOLC criteria");
    } else {
        info!("Proposal rejected by Mercy and/or TOLC criteria");
    }

    (result, acceptable)
}

/// Returns true if the proposal meets both TOLC quality and Mercy Gate standards.
/// This is a lightweight check when you only need the boolean result.
pub fn is_proposal_acceptable(
    model: &llama_cpp_gguf::LlamaModel,
    proposal: &str,
) -> bool {
    evaluate_proposal(model, proposal).is_acceptable()
}
