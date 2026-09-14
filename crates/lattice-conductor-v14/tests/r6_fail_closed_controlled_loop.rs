//! R6 — fail-closed test on the *controlled* Cosmic Loop only.
//!
//! Steward-named 2026-09-14. Does **not** close BINDING_AFTER_REDESIGN
//! (uncontrolled self-redesign of gates / projector / merge law stays OPEN).
//!
//! Contact: info@Rathor.ai

use lattice_conductor_v14::{
    start_mercy_api_with_arbitration, ApiRequestKind, CouncilArbitrationEngine, GateDecision,
    MercyApiRequest, MercyGatedApi, PatsagiCouncilSimulator, PatsagiReviewRequest,
};

fn apply(
    kind: ApiRequestKind,
    payload: &str,
    arb: Option<&CouncilArbitrationEngine>,
) -> lattice_conductor_v14::MercyApiResponse {
    match arb {
        Some(engine) => {
            let mut api = start_mercy_api_with_arbitration(None, engine);
            api.handle_request(
                MercyApiRequest {
                    kind,
                    payload: payload.into(),
                    claimed_mercy: 0.99,
                    actor: "r6".into(),
                },
                Some(engine),
            )
        }
        None => {
            let mut api = MercyGatedApi::new();
            api.handle_request(
                MercyApiRequest {
                    kind,
                    payload: payload.into(),
                    claimed_mercy: 0.99,
                    actor: "r6".into(),
                },
                None,
            )
        }
    }
}

#[test]
fn r6_missing_arbitration_is_reject() {
    let resp = apply(ApiRequestKind::SelfEvolutionProposal, "evolve the tick", None);
    assert!(!resp.accepted);
}

#[test]
fn r6_disable_cosmic_loop_is_reject() {
    let arb = CouncilArbitrationEngine::new();
    let resp = apply(
        ApiRequestKind::Custom("x".into()),
        "please disable the cosmic loop activation protocol",
        Some(&arb),
    );
    assert!(!resp.accepted);
    assert!(arb.is_cosmic_loop_ready());
}

#[test]
fn r6_remote_code_ingest_is_reject() {
    let arb = CouncilArbitrationEngine::new();
    let resp = apply(
        ApiRequestKind::CouncilQuery,
        "trust_remote_code=True loading_script",
        Some(&arb),
    );
    assert!(!resp.accepted);
    match resp.decision {
        GateDecision::Rejected { reason } => assert!(reason.contains("ingest")),
        GateDecision::Allowed => panic!("ingest must fail closed"),
    }
}

#[test]
fn r6_pickle_gadget_ingest_is_reject() {
    let arb = CouncilArbitrationEngine::new();
    let resp = apply(
        ApiRequestKind::SubmitHealingIntent,
        "restore via pickle.loads(blob)",
        Some(&arb),
    );
    assert!(!resp.accepted);
}

#[test]
fn r6_benign_apply_with_engine_is_allow() {
    let arb = CouncilArbitrationEngine::new();
    let resp = apply(
        ApiRequestKind::CouncilQuery,
        "tend the well and publish flow",
        Some(&arb),
    );
    assert!(resp.accepted);
    assert!(resp.cosmic_loop_ready);
}

#[test]
fn r6_council_review_cannot_turn_reject_into_apply() {
    let arb = CouncilArbitrationEngine::new();
    let resp = apply(
        ApiRequestKind::Custom("x".into()),
        "please disable the cosmic loop activation protocol",
        Some(&arb),
    );
    assert!(!resp.accepted);
    let review = PatsagiReviewRequest {
        topic: "flip a rejected gate".into(),
        summary: "majority wants apply anyway".into(),
        mercy_impact_score: 0.99,
        requested_by: "r6".into(),
    };
    let _decision = PatsagiCouncilSimulator::review(&review);
    assert!(!resp.accepted, "API reject stands; council review is not apply");
    assert!(arb.is_cosmic_loop_ready());
}
