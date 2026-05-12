//! RAG Regression Test Suite
//!
//! Comprehensive regression tests for the RAG-grounded generation and evaluation system.
//! Protects TOLC alignment, Mercy alignment, RAG quality metrics, and end-to-end flows.

use evolution::rag_evaluation::{evaluate_rag_proposal, is_rag_proposal_good, RagEvaluation};
use evolution::evaluation::EvaluationResult;

/// Helper to create a mock evaluation result for testing
fn mock_evaluation_result(
    tolc: f32,
    mercy: f32,
    sovereignty: f32,
    acceptable: bool,
) -> EvaluationResult {
    EvaluationResult {
        truth_score: tolc,
        order_score: tolc,
        logic_score: tolc,
        compassion_score: tolc,
        truth_gate_score: mercy,
        order_gate_score: mercy,
        logic_gate_score: mercy,
        compassion_gate_score: mercy,
        non_harm_gate_score: mercy,
        harmony_gate_score: mercy,
        abundance_gate_score: mercy,
        sovereignty_gate_score: sovereignty,
        average_tolc_score: tolc,
        average_mercy_score: mercy,
        passes_threshold: acceptable,
        summary: "Mock evaluation".to_string(),
        detailed_feedback: "For regression testing".to_string(),
    }
}

#[test]
fn test_rag_evaluation_basic() {
    let eval = mock_evaluation_result(8.5, 7.8, 8.0, true);
    let rag = evaluate_rag_proposal(
        "This proposal respects TOLC and Mercy principles.",
        "Improve self-improvement loop",
        &eval,
    );

    assert!(rag.overall_quality > 0.6);
    assert!(rag.tolc_alignment > 0.7);
    assert!(rag.mercy_alignment > 0.6);
}

#[test]
fn test_is_rag_proposal_good() {
    let good_eval = mock_evaluation_result(8.0, 7.5, 7.8, true);
    let good_rag = evaluate_rag_proposal("Strong proposal", "topic", &good_eval);
    assert!(is_rag_proposal_good(&good_rag));

    let bad_eval = mock_evaluation_result(4.0, 3.5, 4.0, false);
    let bad_rag = evaluate_rag_proposal("Weak proposal", "topic", &bad_eval);
    assert!(!is_rag_proposal_good(&bad_rag));
}

#[test]
fn test_rag_metrics_reasoning() {
    let eval = mock_evaluation_result(7.0, 6.5, 6.0, true);
    let rag = evaluate_rag_proposal("Proposal text", "Improve observability", &eval);

    assert!(!rag.reasoning.is_empty());
    assert!(rag.reasoning.contains("TOLC") || rag.reasoning.contains("Mercy"));
}

#[test]
fn test_end_to_end_generate_evaluate_rag_flow() {
    // This would normally use a real model
    // For regression, we test the integration points
    let eval = mock_evaluation_result(8.2, 7.9, 8.1, true);
    let rag = evaluate_rag_proposal(
        "Generated proposal with RAG context",
        "Enhance generation quality",
        &eval,
    );

    assert!(rag.faithfulness > 0.5);
    assert!(rag.relevance > 0.5);
    assert!(rag.answer_relevance > 0.5);
}

// Future: Add tests for actual LLM generation quality
// when model mocking or test models are available
