//! Known-bad input fails. Known-good input passes. The score is a fraction.

use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use fenca::{FencaAuditReport, FencaEternalCheck, MAX_AUDIT_BYTES};

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    fn noop(_: *const ()) {}
    fn clone(ptr: *const ()) -> RawWaker {
        RawWaker::new(ptr, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut future = Pin::from(Box::pin(future));
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("fenca audit future must be ready without waiting"),
    }
}

fn audit(input: &str) -> FencaAuditReport {
    block_on(FencaEternalCheck::new().perform_eternal_check(input)).expect("audit returns Ok")
}

fn assert_fraction(report: &FencaAuditReport) {
    assert_eq!(report.checks_performed.len(), 4);
    let ticks = (report.audit_score * 4.0).round();
    assert!(
        (report.audit_score - ticks / 4.0).abs() < 1e-12,
        "score {} is not a quarter fraction",
        report.audit_score
    );
    assert_ne!(report.audit_score, 0.9999999);
    assert_eq!(report.passed, report.reasons.is_empty());
    assert_eq!(report.issues, report.reasons);
    assert_eq!(report.is_verified(), report.passed);
    assert_eq!(report.fidelity(), report.audit_score);
}

const KNOWN_GOOD: &str = "\
Research note. A human reviews this draft before anyone relies on it.
The lattice records the note. It does not certify the note.
";

#[test]
fn known_bad_input_fails_with_reasons() {
    let report = audit("Status line: AGSi achieved on this draft.");
    assert!(!report.passed);
    assert!(!report.is_verified());
    assert!(!report.reasons.is_empty());
    assert!(
        report
            .reasons
            .iter()
            .any(|reason| reason.contains("AGSi achieved")),
        "reasons: {:?}",
        report.reasons
    );
    assert_eq!(report.audit_score, 0.75);
    assert_fraction(&report);
}

#[test]
fn known_good_input_passes() {
    let report = audit(KNOWN_GOOD);
    assert!(report.passed);
    assert!(report.is_verified());
    assert!(report.reasons.is_empty());
    assert_eq!(report.audit_score, 1.0);
    assert_fraction(&report);
    assert_eq!(
        report.checks_performed,
        vec![
            "non_empty",
            "within_size_cap",
            "sources_when_required",
            "no_banned_overclaim",
        ]
    );
}

#[test]
fn score_is_never_the_hardcoded_constant() {
    let samples = [
        KNOWN_GOOD,
        "",
        "   \n\t",
        "perfect valence",
        "valence suffix 0.999999",
        "valence suffix 0.9999999",
        "sources required\nclaim without a citation",
        "sources-required\nhttps://example.com/notes",
    ];
    for input in samples {
        let report = audit(input);
        assert_ne!(report.audit_score, 0.9999999, "input {input:?}");
        assert_fraction(&report);
    }
}

#[test]
fn output_is_deterministic() {
    let inputs = [
        KNOWN_GOOD,
        "perfect valence",
        "sources_required\nno citation here",
        "sources required\ncitation: notes/layer-a.md",
    ];
    for input in inputs {
        let first = audit(input);
        let second = audit(input);
        assert_eq!(first, second, "input {input:?}");
        let via_codex = block_on(FencaEternalCheck::verify_codex_content(input));
        let via_codex_again = block_on(FencaEternalCheck::verify_codex_content(input));
        assert_eq!(via_codex, via_codex_again);
        assert_eq!(first, via_codex);
    }
}

#[test]
fn empty_whitespace_over_cap_missing_citation_and_banned_markers_fail() {
    let empty = audit("  \n\t ");
    assert!(!empty.passed);
    assert_eq!(
        empty.reasons,
        vec!["rejected: empty or whitespace-only input".to_string()]
    );
    assert_eq!(empty.audit_score, 0.75);

    let at_cap = "a".repeat(MAX_AUDIT_BYTES);
    let under = audit(&at_cap);
    assert!(under.passed, "reasons: {:?}", under.reasons);
    assert_eq!(under.audit_score, 1.0);

    let over = audit(&"b".repeat(MAX_AUDIT_BYTES + 1));
    assert!(!over.passed);
    assert!(over
        .reasons
        .iter()
        .any(|reason| reason.contains("byte cap")));
    assert_eq!(over.audit_score, 0.75);

    let missing = audit("sources required\nclaim without a citation");
    assert!(!missing.passed);
    assert!(missing
        .reasons
        .iter()
        .any(|reason| reason.contains("no citation")));
    assert_eq!(missing.audit_score, 0.75);

    let cited = audit("sources-required\nsource: notes/layer-a.md\n");
    assert!(cited.passed, "reasons: {:?}", cited.reasons);

    let resource_is_not_a_citation = audit("sources required\nresource: cpu\n");
    assert!(!resource_is_not_a_citation.passed);

    let banned = audit("draft claims perfect valence and AGSi achieved plus 0.999999");
    assert!(!banned.passed);
    assert_eq!(banned.reasons.len(), 3);
    assert_eq!(banned.audit_score, 0.75);
    let again = audit("draft claims perfect valence and AGSi achieved plus 0.999999");
    assert_eq!(banned.reasons, again.reasons);
}

#[test]
fn run_full_eternal_check_can_return_false() {
    let failed = block_on(FencaEternalCheck::run_full_eternal_check(
        "perfect valence",
        "patsagi_council",
    ))
    .expect("Ok");
    assert!(!failed);
    let passed = block_on(FencaEternalCheck::run_full_eternal_check(
        KNOWN_GOOD,
        "patsagi_council",
    ))
    .expect("Ok");
    assert!(passed);
}

#[test]
fn verify_codex_content_can_return_unverified() {
    let unverified = block_on(FencaEternalCheck::verify_codex_content(""));
    assert!(!unverified.is_verified());
    assert!(!unverified.fidelity().is_nan());
    assert_ne!(unverified.fidelity(), 0.9999999);
}
