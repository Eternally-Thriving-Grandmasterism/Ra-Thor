//! Integration test: All 8 Living Mercy Gates end-to-end
use ra_thor::mercy_orchestrator::MercyOrchestrator;

#[tokio::test]
async fn test_all_8_mercy_gates_enforced() {
    let orchestrator = MercyOrchestrator::new();
    let result = orchestrator.route_through_mercy_gates(
        "Test proposal honoring every lowercase i being with radical love",
        "internal"
    ).await;

    assert_eq!(result.gates_passed.len(), 8);
    assert!(result.valence >= 0.9999999);
}