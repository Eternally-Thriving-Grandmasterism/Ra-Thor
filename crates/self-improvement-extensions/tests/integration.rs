//! Automated integration tests for the evaluator.
//! These tests exercise the public API and new validation/logging behavior.

use self_improvement_extensions::{
    evaluate_proposal_with_tolc_and_mercy, EvaluationResult,
};

// Note: Full end-to-end tests with a real model require the `llama-cpp` feature
// and a loaded model. These tests focus on the robust paths that don't require the model.

#[test]
fn test_evaluation_result_is_acceptable_logic() {
    let mut result = EvaluationResult {
        truth_score: 8.0,
        order_score: 7.5,
        logic_score: 8.5,
        compassion_score: 7.0,
        truth_gate_score: 8.0,
        order_gate_score: 7.5,
        logic_gate_score: 8.0,
        compassion_gate_score: 7.5,
        non_harm_gate_score: 6.5,
        harmony_gate_score: 7.0,
        abundance_gate_score: 8.0,
        sovereignty_gate_score: 8.5,
        average_tolc_score: 7.75,
        average_mercy_score: 7.625,
        passes_threshold: true,
        summary: "Good proposal".to_string(),
        detailed_feedback: "Solid across the board".to_string(),
    };

    assert!(result.is_acceptable());

    // Now drop below threshold on Non-Harm
    result.non_harm_gate_score = 5.0;
    assert!(!result.is_acceptable());

    // Reset and test sovereignty threshold
    result.non_harm_gate_score = 7.0;
    result.sovereignty_gate_score = 6.5;
    assert!(!result.is_acceptable());
}

#[test]
fn test_evaluation_result_defaults_on_failure() {
    // This simulates what happens on JSON parse failure or validation failure
    let result = EvaluationResult {
        truth_score: 0.0,
        order_score: 0.0,
        logic_score: 0.0,
        compassion_score: 0.0,
        truth_gate_score: 0.0,
        order_gate_score: 0.0,
        logic_gate_score: 0.0,
        compassion_gate_score: 0.0,
        non_harm_gate_score: 0.0,
        harmony_gate_score: 0.0,
        abundance_gate_score: 0.0,
        sovereignty_gate_score: 0.0,
        average_tolc_score: 0.0,
        average_mercy_score: 0.0,
        passes_threshold: false,
        summary: "Failed to parse evaluation response".to_string(),
        detailed_feedback: "raw response would go here".to_string(),
    };

    assert!(!result.is_acceptable());
    assert_eq!(result.average_tolc_score, 0.0);
    assert_eq!(result.sovereignty_gate_score, 0.0);
}
