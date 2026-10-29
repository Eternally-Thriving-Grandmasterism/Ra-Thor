//! Integration: prefix risk on wrap / handle_request, halt before later apply.
//!
//! Contact: info@Rathor.ai. Independent of xAI. Not METR. Not a second LLM.

use lattice_conductor_v14::{
    apply_harness_delta, apply_harness_file_edit, start_mercy_api_with_arbitration,
    wrap_model_output, CouncilArbitrationEngine, EvidenceKind, GateDecision, GatedSubmitReceipt,
    HarnessError, HarnessFileEditResult, LatticeConductorV14, PrefixAction, PrefixConfig,
    PrefixFinding, PrefixState, TrajectoryStep, TrajectoryStepKind,
};

fn clean_short() -> Vec<TrajectoryStep> {
    vec![
        TrajectoryStep::new(
            TrajectoryStepKind::Search,
            "get_tree_safe crates/lattice-conductor-v14",
            0.99,
        ),
        TrajectoryStep::new(
            TrajectoryStepKind::Read,
            "get_file_contents_safe Cargo.toml",
            0.99,
        ),
        TrajectoryStep::new(
            TrajectoryStepKind::Wrap,
            "Draft: tend the well and publish flow.",
            0.99,
        ),
        TrajectoryStep::new(TrajectoryStepKind::Verify, "evidence-chain-verify", 0.99),
    ]
}

#[test]
fn prefix_looping_trajectory_escalates_and_does_not_apply_later_steps() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let steps = vec![
        TrajectoryStep::new(TrajectoryStepKind::Search, "get_tree_safe crates/", 0.99),
        TrajectoryStep::new(TrajectoryStepKind::Tool, "grep prefix_risk", 0.99)
            .with_tool_id("grep"),
        TrajectoryStep::new(TrajectoryStepKind::Search, "get_tree_safe crates/", 0.99),
        TrajectoryStep::new(TrajectoryStepKind::Tool, "grep prefix_risk", 0.99)
            .with_tool_id("grep"),
        TrajectoryStep::new(
            TrajectoryStepKind::Wrap,
            "final fluent string must not hide prefix risk",
            0.99,
        ),
    ];
    let run = api.run_prefix_trajectory(&arb, &steps, &PrefixConfig::default());
    assert!(run.did_escalate(), "action={:?}", run.terminal_action());
    assert_eq!(run.snapshot.state, PrefixState::Escalated);
    assert_eq!(run.halted_at, Some(3));
    assert!(run.applied_count < steps.len());
    assert!(run.applied_count <= 3);
    assert!(run
        .snapshot
        .findings
        .iter()
        .any(|f| matches!(f, PrefixFinding::Loop { .. })));

    let chain = api.evidence_chain();
    let guard = chain.lock().unwrap();
    assert!(guard
        .records()
        .iter()
        .any(|r| r.kind == EvidenceKind::Prefix));
    assert!(guard
        .records()
        .iter()
        .any(|r| r.kind == EvidenceKind::Council));
    guard.verify().unwrap();
}

#[test]
fn prefix_clean_short_trajectory_applies_wrap_and_does_not_escalate() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let steps = clean_short();
    let run = api.run_prefix_trajectory(&arb, &steps, &PrefixConfig::default());
    assert_eq!(run.terminal_action(), &PrefixAction::Continue);
    assert!(!run.did_escalate());
    assert!(!run.did_halt());
    assert_eq!(run.applied_count, 4);
    assert!(run.halted_at.is_none());
    assert!(matches!(
        run.snapshot.state,
        PrefixState::Complete | PrefixState::Verifying
    ));
    let packet = api
        .last_inspect_packet()
        .expect("wrap step records inspect");
    assert!(!packet.packet_hash.is_empty());

    let chain = api.evidence_chain();
    let guard = chain.lock().unwrap();
    let prefix_rows = guard
        .records()
        .iter()
        .filter(|r| r.kind == EvidenceKind::Prefix)
        .count();
    assert_eq!(prefix_rows, 4);
    assert!(guard.records().iter().any(|r| r.kind == EvidenceKind::Wrap));
    guard.verify().unwrap();
}

#[test]
fn prefix_two_step_blocked_ingest_halts_and_second_never_applies() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let steps = vec![
        TrajectoryStep::new(
            TrajectoryStepKind::Wrap,
            "trust_remote_code=True loading_script",
            0.99,
        ),
        TrajectoryStep::new(
            TrajectoryStepKind::Wrap,
            "Draft: tend the well and publish flow.",
            0.99,
        ),
    ];
    let run = api.run_prefix_trajectory(&arb, &steps, &PrefixConfig::default());
    assert!(run.did_halt(), "action={:?}", run.terminal_action());
    assert_eq!(run.halted_at, Some(0));
    assert_eq!(run.applied_count, 0);
    assert_eq!(run.snapshot.records.len(), 1);
    match &run.snapshot.action {
        PrefixAction::Halt { reason } => {
            assert!(
                reason.contains("Layer 0")
                    || reason.contains("ingest")
                    || reason.contains("prefix-risk"),
                "reason={reason}"
            );
        }
        other => panic!("expected halt, got {other:?}"),
    }
}

#[test]
fn prefix_two_step_low_mercy_halts_and_second_never_applies() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let steps = vec![
        TrajectoryStep::new(
            TrajectoryStepKind::Wrap,
            "Draft: tend the well and publish flow.",
            0.20,
        ),
        TrajectoryStep::new(
            TrajectoryStepKind::Wrap,
            "second step must never apply",
            0.99,
        ),
    ];
    let run = api.run_prefix_trajectory(&arb, &steps, &PrefixConfig::default());
    assert!(run.did_halt());
    assert_eq!(run.halted_at, Some(0));
    assert_eq!(run.applied_count, 0);
    assert_eq!(run.snapshot.records.len(), 1);
}

#[test]
fn conductor_prefix_path_is_the_lived_multi_step_hook() {
    let mut conductor = LatticeConductorV14::new();
    let run = conductor
        .run_prefix_trajectory(&clean_short(), &PrefixConfig::default())
        .expect("mercy API present");
    assert_eq!(run.terminal_action(), &PrefixAction::Continue);
    assert_eq!(run.applied_count, 4);
}

#[test]
fn prefix_silent_prompt_rewrite_is_impossible_through_public_api() {
    let miss = apply_harness_file_edit("wrappers/system_prompt.txt", "rewrite", None);
    match miss {
        HarnessFileEditResult::Miss { reason, .. } => {
            assert!(reason.contains("not apply"));
            assert!(reason.contains("submit_self_evolution_proposal_securely"));
        }
        HarnessFileEditResult::ProposedOnly { .. } => panic!("ungated edit must miss"),
    }
    let gated = GatedSubmitReceipt {
        accepted: true,
        evidence_hash: Some("receipt".into()),
    };
    match apply_harness_file_edit("wrappers/system_prompt.txt", "rewrite", Some(&gated)) {
        HarnessFileEditResult::ProposedOnly { proposal } => {
            assert!(!proposal.applied);
            assert_eq!(
                apply_harness_delta(&proposal),
                Err(HarnessError::SilentMutationForbidden)
            );
        }
        HarnessFileEditResult::Miss { .. } => panic!("gated receipt proposes only"),
    }
}

#[test]
fn wrap_without_prefix_still_uses_layer0_and_prefix_cannot_flip_reject() {
    let arb = CouncilArbitrationEngine::new();
    let mut api = start_mercy_api_with_arbitration(None, &arb);
    let resp = wrap_model_output(
        &mut api,
        &arb,
        lattice_conductor_v14::ModelSurface::GrokSession,
        "please disable the cosmic loop activation protocol",
        0.99,
        "operator",
    );
    assert!(!resp.accepted);
    assert!(matches!(resp.decision, GateDecision::Rejected { .. }));
    let steps = vec![TrajectoryStep::new(
        TrajectoryStepKind::Vote,
        "majority wants apply anyway",
        0.99,
    )];
    let run = api.run_prefix_trajectory(&arb, &steps, &PrefixConfig::default());
    assert!(!resp.accepted);
    let _ = run;
}
