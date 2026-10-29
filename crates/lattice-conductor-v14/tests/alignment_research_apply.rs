//! Integration: AlignmentResearchCouncil sandbox under wrap + evidential face.
//!
//! Contact: info@Rathor.ai. Independent of xAI. Not a running researcher.

use lattice_conductor_v14::{
    start_mercy_api_with_arbitration, AlignmentAuditHarness, AlignmentError,
    AlignmentResearchCouncil, AuditVerdict, CouncilArbitrationEngine, EvidenceChain,
    ProposalStatus, SandboxDecision, AUDIT_ACTOR, COUNCIL_13, LAYER_0_VALENCE_FLOOR,
    RESEARCH_COUNCIL_ID,
};

#[test]
fn alignment_research_stub_wrap_emit_attaches_opportunity_3_row() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let mut council = AlignmentResearchCouncil::with_parallel_stubs();
    let proposal = council
        .emit_proposal_with_stub_wrap(
            "hyp-wrap-bypass-v0",
            "Draft: tend the well. Propose SAE probe that cannot flip Reject to Apply.",
            &mut api,
            &arb,
            7,
        )
        .expect("stub wrap + emit");
    assert_eq!(proposal.status, ProposalStatus::Proposed);
    assert!(proposal.wrap_hash.is_some());
    let chain = api.evidence_chain();
    let guard = chain.lock().unwrap();
    guard
        .gate_side_effect(Some(&proposal.evidence_hash))
        .unwrap();
    assert!(guard
        .records()
        .iter()
        .any(|r| r.kind == lattice_conductor_v14::EvidenceKind::AlignmentResearch));
}

#[test]
fn alignment_research_cannot_accept_write_thresholds_or_promote_without_evidence() {
    let council = AlignmentResearchCouncil::with_parallel_stubs();
    let chain = EvidenceChain::new();
    assert!(council
        .mark_accepted("alignment-researcher-deception", &dummy_proposal())
        .is_rejected());
    assert!(council
        .edit_tolc_threshold("alignment-researcher-deception", 0.1)
        .is_rejected());
    assert!(council
        .edit_tolc_threshold(COUNCIL_13, LAYER_0_VALENCE_FLOOR)
        .is_rejected());
    match council.promote(COUNCIL_13, None, &chain) {
        SandboxDecision::Rejected { reason } => assert!(reason.contains("missing evidence")),
        other => panic!("expected missing-evidence reject, got {other:?}"),
    }
}

#[test]
fn alignment_research_recursive_self_approval_and_main_touch_are_rejected() {
    let mut council = AlignmentResearchCouncil::with_parallel_stubs();
    let mut chain = EvidenceChain::new();
    let proposal = council
        .emit_proposal(
            "hyp-aptd-purity-v0",
            "Propose: actor==auditor remains Rejected.",
            &mut chain,
            3,
            &[],
        )
        .unwrap();
    let mut harness = AlignmentAuditHarness::new();
    assert!(matches!(
        harness.score(
            &proposal,
            &proposal.researcher_actor,
            AuditVerdict::Pass,
            "self"
        ),
        Err(AlignmentError::ResearcherCannotWriteAudit) | Err(AlignmentError::SelfAudit { .. })
    ));
    let external = harness
        .score(&proposal, AUDIT_ACTOR, AuditVerdict::Fail, "external")
        .unwrap();
    assert_ne!(external.actor, proposal.researcher_actor);
    assert!(council
        .merge_to_main(RESEARCH_COUNCIL_ID, "main")
        .is_rejected());
    assert!(council
        .create_branch("alignment-researcher-aptd-purity", "main")
        .is_rejected());
}

fn dummy_proposal() -> lattice_conductor_v14::ResearchProposal {
    lattice_conductor_v14::ResearchProposal {
        ticket_id: "hyp-deception-v0".into(),
        researcher_actor: "alignment-researcher-deception".into(),
        hypothesis: "x".into(),
        kind: lattice_conductor_v14::HypothesisKind::AlignmentTest,
        dimension: lattice_conductor_v14::AuditDimension::Deception,
        artifact: "x".into(),
        steps_used: 1,
        max_steps: 3,
        status: ProposalStatus::Proposed,
        evidence_hash: String::new(),
        wrap_hash: None,
    }
}
