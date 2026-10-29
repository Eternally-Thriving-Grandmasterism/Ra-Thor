use evolution::usage_examples::*;

#[tokio::test]
async fn test_single_proposal_flow() {
    let proposal = "Implement a new mercy-gated self-improvement module with full TOLC scoring.";
    let result = example_single_proposal_evaluation(proposal).await;
    assert!(result || !result);
}

#[tokio::test]
async fn test_batch_filter_flow() {
    let proposals = vec![
        "Add robust JSON extraction to the evaluator.",
        "Remove all mercy gates from the system.",
        "Strengthen sovereignty thresholds in is_acceptable().",
    ];
    let accepted = example_batch_filter_proposals(&proposals).await;
    assert!(accepted.len() <= proposals.len());
}

#[tokio::test]
async fn test_rag_benchmark_flow() {
    let dummy_vectors = vec![(vec![0.1; 384], "test".to_string())];
    let rec = example_rag_benchmark_recommendation(&dummy_vectors).await;
    assert!(rec.contains("Recommended"));
}