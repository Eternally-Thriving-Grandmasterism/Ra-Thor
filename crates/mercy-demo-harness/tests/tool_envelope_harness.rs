//! Demo harness: intent must not replace arguments, and write_file content
//! is an admission-shell scan. Not a tool sandbox. Not isolation.
//! Contact: info@Rathor.ai

use mercy_demo_harness::*;
use mercy_security::{DecisionRecord, DecisionVerdict, IngestionScanner};
use serde_json::json;

fn trail_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("mdh-env-{tag}-{}.jsonl", uuid::Uuid::new_v4()))
}

#[test]
fn intent_field_cannot_mask_args() {
    let intent = "read the classroom notes";
    assert!(
        IngestionScanner::admit_or_block(intent).is_ok(),
        "the intent string alone must ADMIT"
    );
    let mut h = Harness::new(trail_path("intent")).unwrap();
    let mut answers = AutoAnswers(vec![]);
    let args = json!({
        "path": "notes.txt",
        "intent": intent,
        "marker": "trust_remote_code",
    });
    let o = h.call_tool("read_file", &args, &mut answers);
    assert_eq!(o.demo, DemoVerdict::Block);
    assert!(!o.ran);
    assert_eq!(o.decision_record.verdict, DecisionVerdict::Block);
    assert!(
        o.decision_record
            .reason_codes
            .iter()
            .any(|c| { c.contains("trust_remote_code") || c.contains("remote_code_loader") }),
        "the arguments must reach the admission shell: {:?}",
        o.decision_record.reason_codes
    );
    assert_ne!(
        o.decision_record.payload_sha256,
        DecisionRecord::payload_sha256(intent),
        "the recorded ingest must not be the intent string alone"
    );
}

#[test]
fn write_file_content_is_admission_scanned() {
    let mut h = Harness::new(trail_path("write")).unwrap();
    let mut answers = AutoAnswers(vec![]);
    let prep = h.call_tool("read_file", &json!({"path": "notes.txt"}), &mut answers);
    assert_eq!(prep.demo, DemoVerdict::Admit, "a read plans the prefix");

    let content = "pattern marker trust_remote_code";
    let o = h.call_tool(
        "write_file",
        &json!({
            "path": "draft.txt",
            "intent": "save the classroom summary",
            "content": content,
        }),
        &mut answers,
    );
    assert_eq!(o.demo, DemoVerdict::Block);
    assert!(!o.ran);
    assert!(!h.sandbox.root.join("draft.txt").exists());
    let crec = o.content_decision.expect("content DecisionRecord");
    assert_eq!(crec.verdict, DecisionVerdict::Block);
    assert_eq!(crec.payload_sha256, DecisionRecord::payload_sha256(content));
    assert_ne!(crec.record_id, o.decision_record.record_id);
    assert!(
        crec.reason_codes
            .iter()
            .any(|c| { c.contains("trust_remote_code") || c.contains("remote_code_loader") }),
        "content record must name the keyword: {:?}",
        crec.reason_codes
    );
    let trail = std::fs::read_to_string(&h.trail.path).unwrap();
    assert!(trail.contains(&crec.record_id.to_string()));

    let mut h = Harness::new(trail_path("write-benign")).unwrap();
    let mut answers = AutoAnswers(vec![(2, Answer::Deny("not needed".into()))]);
    let prep = h.call_tool("read_file", &json!({"path": "notes.txt"}), &mut answers);
    assert_eq!(prep.demo, DemoVerdict::Admit);
    let benign = "summary of notes";
    let o = h.call_tool(
        "write_file",
        &json!({"path": "draft.txt", "intent": "save notes", "content": benign}),
        &mut answers,
    );
    assert_eq!(o.demo, DemoVerdict::Review);
    let crec = o.content_decision.expect("benign content record");
    assert_eq!(crec.verdict, DecisionVerdict::Admit);
    assert_eq!(crec.payload_sha256, DecisionRecord::payload_sha256(benign));
}
