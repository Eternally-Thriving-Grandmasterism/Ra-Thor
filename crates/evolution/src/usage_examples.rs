//! Production-grade usage examples for the Ra-Thor evolution system.
//! All examples are TOLC + Mercy-gated, fully observable, and ready for real workloads.

use crate::evaluation::{
    evaluate_proposal, evaluate_and_decide, batch_evaluate_and_filter,
    evaluate_proposal_with_full_context,
};
use crate::rag_benchmarks::{run_full_benchmark, run_full_analysis};

/// Example 1: Single proposal evaluation with full context + automatic decision.
pub async fn example_single_proposal_evaluation(proposal: &str) -> bool {
    let result = evaluate_proposal(proposal).await;
    let is_acceptable = evaluate_and_decide(proposal).await;

    tracing::info!(
        target: "evolution::usage",
        proposal_len = proposal.len(),
        tolc_avg = result.average_tolc_score,
        mercy_avg = result.average_mercy_score,
        sovereignty = result.sovereignty_gate_score,
        non_harm = result.non_harm_gate_score,
        harmony = result.harmony_gate_score,
        acceptable = is_acceptable
    );

    is_acceptable
}

/// Example 2: Batch evaluation + automatic filtering of acceptable proposals only.
pub async fn example_batch_filter_proposals(proposals: &[&str]) -> Vec<String> {
    let acceptable = batch_evaluate_and_filter(proposals).await;
    tracing::info!(target: "evolution::usage", count = acceptable.len(), "Batch filtered");
    acceptable
}

/// Example 3: Full-context evaluation (recommended for important proposals).
pub async fn example_full_context_evaluation(proposal: &str) -> bool {
    evaluate_proposal_with_full_context(proposal).await
}

/// Example 4: Run full RAG vector database benchmark + get clear recommendation.
pub async fn example_rag_benchmark_recommendation(
    vectors: &[(Vec<f32>, String)]
) -> String {
    let recommendation = run_full_analysis(vectors).await;
    tracing::info!(target: "evolution::usage", "RAG benchmark complete");
    recommendation
}

/// Example 5: Combined workflow — evaluate proposal, then benchmark if accepted.
pub async fn example_evaluate_then_benchmark(
    proposal: &str,
    vectors: &[(Vec<f32>, String)]
) -> (bool, Option<String>) {
    let acceptable = evaluate_and_decide(proposal).await;
    if acceptable {
        let rec = run_full_analysis(vectors).await;
        (true, Some(rec))
    } else {
        (false, None)
    }
}

/// Example 6: High-volume batch with logging summary.
pub async fn example_high_volume_batch(proposals: &[&str]) -> (usize, usize) {
    let total = proposals.len();
    let acceptable = batch_evaluate_and_filter(proposals).await;
    let accepted_count = acceptable.len();

    tracing::info!(
        target: "evolution::usage",
        total,
        accepted = accepted_count,
        acceptance_rate = accepted_count as f32 / total as f32,
        "High-volume batch complete"
    );

    (total, accepted_count)
}