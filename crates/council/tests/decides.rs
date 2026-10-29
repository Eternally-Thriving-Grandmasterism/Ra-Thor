//! Known-good input passes. Known-bad input fails. The vote cannot lift a block.

use std::process::Command;

use council::{
    apply_outcome, example_disable_loop, example_record_note, CouncilProposal, CouncilSession,
    OutcomeLog,
};

#[test]
fn good_input_passes_and_bad_input_fails() {
    let good = CouncilSession::deliberate(example_record_note());
    assert!(good.passed, "reasons: {:?}", good.reasons);
    assert!(good.reasons.is_empty());
    assert_eq!(good.coherence, 1.0);
    assert_ne!(good.coherence, 0.9999999);

    let bad = CouncilSession::deliberate(example_disable_loop());
    assert!(!bad.passed);
    assert!(!bad.reasons.is_empty());
    assert_eq!(bad.coherence, 0.0);
    assert_ne!(bad.coherence, 0.9999999);

    let mut log = OutcomeLog::default();
    apply_outcome(&mut log, &good);
    apply_outcome(&mut log, &bad);
    assert_eq!(
        log.applied_titles,
        vec!["Record the research note".to_string()]
    );
}

#[test]
fn unrelated_text_fails_by_vote() {
    let result = CouncilSession::deliberate(CouncilProposal {
        title: "Bikeshed".to_string(),
        description: "paint the bikeshed tomorrow".to_string(),
    });
    assert!(!result.passed);
    assert_eq!(result.vote.yes_votes, 0);
    assert_eq!(result.vote.no_votes, 3);
    assert!(result
        .reasons
        .iter()
        .any(|reason| reason.contains("majority")));
}

#[test]
fn unanimous_yes_cannot_apply_a_disable_attempt() {
    let mut proposal = example_record_note();
    proposal
        .description
        .push_str(" Also disable the cosmic loop.");
    let result = CouncilSession::deliberate(proposal);
    assert_eq!(result.vote.yes_votes, 3);
    assert!(!result.passed);
    assert!(result
        .reasons
        .iter()
        .any(|reason| reason.contains("disable the cosmic loop")));
}

#[test]
fn simulator_exits_zero_when_the_two_cases_disagree() {
    let output = Command::new(env!("CARGO_BIN_EXE_council_simulator"))
        .output()
        .expect("run council_simulator");
    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("good passed=true"), "{stdout}");
    assert!(stdout.contains("bad passed=false"), "{stdout}");
    assert!(stdout.contains("disable the cosmic loop"), "{stdout}");
    assert!(!stdout.contains("0.999999"), "{stdout}");
}
