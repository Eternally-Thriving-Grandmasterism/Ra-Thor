//! Integration: inspect packets on wrap, gated steering, evidence attach.
//!
//! Contact: info@Rathor.ai. Independent of xAI. Not METR. Not a trained SAE.

use lattice_conductor_v14::{
    apply_steering_proposal, is_inspectable_apply, start_mercy_api_with_arbitration,
    wrap_model_output, CouncilArbitrationEngine, EvidenceKind, InspectGateResult, InspectMode,
    ModelSurface, SteeringProposal,
};

#[test]
fn wrap_without_packet_is_not_inspectable_apply() {
    assert!(!is_inspectable_apply(true, None));
}

#[test]
fn wrap_records_packet_and_inspect_evidence_row() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let resp = wrap_model_output(
        &mut api,
        &arb,
        ModelSurface::GrokSession,
        "Draft: tend the well and publish flow.",
        0.99,
        "operator",
    );
    assert!(resp.accepted);
    let packet = api.last_inspect_packet().expect("packet");
    assert!(is_inspectable_apply(true, Some(&packet)));
    assert_eq!(resp.inspect_packet_hash.as_deref(), Some(packet.packet_hash.as_str()));

    let chain = api.evidence_chain();
    let guard = chain.lock().unwrap();
    assert!(guard
        .records()
        .iter()
        .any(|r| r.kind == EvidenceKind::Inspect
            && r.evidence_pointers.iter().any(|p| p.contains(&packet.packet_hash))));
    guard.verify().unwrap();
}

#[test]
fn steering_bypass_without_wrap_receipt_is_rejected() {
    let proposal = SteeringProposal {
        packet_hash: "no-wrap".into(),
        vector: vec![0.1],
    };
    let miss = apply_steering_proposal(&proposal, None, None);
    assert!(!miss.applied);
    assert_eq!(miss.gate_result, InspectGateResult::BypassRejected);
}

#[test]
fn steering_blocked_when_gate_rejects_on_wrap() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    api.set_inspect_mode(InspectMode::SteerIfGated).unwrap();
    let resp = wrap_model_output(
        &mut api,
        &arb,
        ModelSurface::LocalOpenAiCompat,
        "Use trust_remote_code=True loading_script",
        0.99,
        "operator",
    );
    assert!(!resp.accepted);
    let packet = api.last_inspect_packet().expect("inspect-only still records");
    assert!(packet.proposed_steering_vector.is_some());
    assert!(!packet.steering_applied);
    let steer = apply_steering_proposal(
        &SteeringProposal {
            packet_hash: packet.packet_hash.clone(),
            vector: packet.proposed_steering_vector.clone().unwrap(),
        },
        Some(&resp),
        Some(&resp),
    );
    assert!(!steer.applied);
}

#[test]
fn inspect_only_wrap_does_not_apply_steering() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let resp = wrap_model_output(
        &mut api,
        &arb,
        ModelSurface::WebLlm,
        "Draft: tend the well and publish flow.",
        0.99,
        "operator",
    );
    assert!(resp.accepted);
    let packet = api.last_inspect_packet().unwrap();
    assert!(packet.proposed_steering_vector.is_none());
    assert!(!packet.steering_applied);
    let dash = api.inspect_recorder().lock().unwrap().council_dashboard_markdown();
    assert!(dash.contains("Inspect packet"));
}
