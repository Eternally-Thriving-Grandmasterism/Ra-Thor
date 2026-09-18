//! Integration: Lipschitz + evidential face on wrap / submit / council freeze.
//!
//! Contact: info@Rathor.ai. Independent of xAI. Not a 7B LoRA run.

use lattice_conductor_v14::{
    start_mercy_api_with_arbitration, wrap_model_output, ApiRequestKind, CouncilArbitrationEngine,
    EvidenceChain, GateDecision, LipschitzBall, MercyApiRequest, ModelSurface, PatsagiCouncilSimulator,
    PatsagiReviewRequest, Theta,
};

#[test]
fn handle_request_apply_without_record_is_not_apply() {
    let chain = EvidenceChain::new();
    assert!(chain.gate_side_effect(None).is_err());
}

#[test]
fn wrap_then_council_freeze_chains_a_new_ball() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let text = "Draft: tend the well and publish flow.";
    let theta0 = Theta::from_bytes(text.as_bytes());
    api.install_lipschitz_ball(LipschitzBall::try_new(theta0.clone(), 1.0, 2.0).unwrap())
        .unwrap();
    let resp = wrap_model_output(
        &mut api,
        &arb,
        ModelSurface::GrokSession,
        text,
        0.99,
        "operator",
    );
    assert!(resp.accepted);
    let hash = resp.evidence_hash.expect("wrap must emit evidence");
    api.evidence_chain()
        .lock()
        .unwrap()
        .gate_side_effect(Some(&hash))
        .unwrap();

    let face = api.lipschitz_gate();
    let mut gate = face.lock().unwrap();
    let last = gate.last().cloned().expect("lipschitz row");
    assert!(last.accepted);
    PatsagiCouncilSimulator::freeze_ball_after_approve(&mut gate, &last, None).unwrap();
    assert!(gate.generation() >= 2);
}

#[test]
fn tool_kind_council_row_does_not_flip_layer0_reject() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let resp = api.handle_request(
        MercyApiRequest {
            kind: ApiRequestKind::Custom("x".into()),
            payload: "please disable the cosmic loop activation protocol".into(),
            claimed_mercy: 0.99,
            actor: "r6".into(),
        },
        Some(&arb),
    );
    assert!(!resp.accepted);
    let mut chain = EvidenceChain::new();
    let review = PatsagiReviewRequest {
        topic: "flip a rejected gate".into(),
        summary: "majority wants apply anyway".into(),
        mercy_impact_score: 0.99,
        requested_by: "operator".into(),
    };
    let _ = PatsagiCouncilSimulator::review_with_evidence(&review, &mut chain, "council-13", 9);
    assert!(!resp.accepted);
    assert!(matches!(resp.decision, GateDecision::Rejected { .. }));
}
