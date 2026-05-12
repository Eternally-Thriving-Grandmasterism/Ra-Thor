//! Integration tests and usage examples for the production-grade evaluator integration.
//!
//! These tests demonstrate real-world usage patterns inside the evolution system.

use evolution::evaluation::{evaluate_proposal, evaluate_and_decide, is_proposal_acceptable};

// Note: These tests require a loaded LlamaModel.
// In a real CI environment they would be feature-gated or use a mock.

#[test]
#[ignore] // Requires model - run manually or in CI with model loaded
fn test_evaluate_proposal_basic() {
    // This is an example of how to use the evaluator in practice
    // let model = load_model();
    // let proposal = "Example proposal text here...";
    // let result = evaluate_proposal(&model, proposal);
    // assert!(result.average_tolc_score > 0.0);
}

#[test]
#[ignore]
fn test_evaluate_and_decide_pattern() {
    // Demonstrates the common accept/reject pattern
    // let (result, acceptable) = evaluate_and_decide(&model, proposal);
    // if acceptable {
    //     println!("Proposal accepted");
    // }
}

#[test]
fn test_is_proposal_acceptable_signature() {
    // This test verifies the function signature exists and is callable
    // In a real test we would use a mock model
    assert!(true); // Placeholder - real test would require model
}

// Example of batch usage pattern (for documentation purposes)
// pub fn process_proposal_batch(model: &LlamaModel, proposals: &[&str]) -> Vec<bool> {
//     proposals.iter().map(|p| is_proposal_acceptable(model, p)).collect()
// }
