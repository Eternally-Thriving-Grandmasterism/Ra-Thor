//! Decision-record schema lock — public fixtures, JSON round-trip, no raw payload.
//!
//! Not a hash-chained SIEM. Not EU AI Act logging. inspect ≠ METR.
//! Contact: info@Rathor.ai

use std::fs;
use std::path::PathBuf;

use mercy_security::{
    DecisionActor, DecisionRecord, DecisionRecordError, DecisionThreatClass, DecisionVerdict,
    IngestionScanner, DECISION_RECORD_POLICY_VERSION, DECISION_RECORD_SURFACE,
};

fn public_fixture(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/mercy-security")
        .join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

#[test]
fn admitted_public_fixture_emits_admit_record() {
    let text = public_fixture("benign/model_card_clean.md");
    assert!(
        IngestionScanner::admit_or_block(&text).is_ok(),
        "model_card_clean.md must still ADMIT under living thresholds"
    );
    let rec = DecisionRecord::from_unattended_ingest(&text);
    assert_eq!(rec.verdict, DecisionVerdict::Admit);
    assert_eq!(rec.actor, DecisionActor::Unattended);
    assert_eq!(rec.surface, DECISION_RECORD_SURFACE);
    assert_eq!(rec.policy_version, DECISION_RECORD_POLICY_VERSION);
    assert_eq!(rec.crate_version, env!("CARGO_PKG_VERSION"));
    assert!(matches!(
        rec.threat_class,
        DecisionThreatClass::None | DecisionThreatClass::Low
    ));
    assert_eq!(rec.payload_sha256, DecisionRecord::payload_sha256(&text));
    assert_eq!(rec.payload_sha256.len(), 64);
    assert!(rec.override_rationale.is_none());
    assert!(rec.prev_verdict.is_none());
    assert!(rec.reason_codes.iter().any(|c| c.starts_with("tier:")));
}

#[test]
fn blocked_public_fixture_emits_block_record() {
    let text = public_fixture("blocked/trust_remote_code_loader.txt");
    assert!(
        IngestionScanner::admit_or_block(&text).is_err(),
        "trust_remote_code_loader.txt must still BLOCK under living thresholds"
    );
    let rec = DecisionRecord::from_unattended_ingest(&text);
    assert_eq!(rec.verdict, DecisionVerdict::Block);
    assert_eq!(rec.actor, DecisionActor::Unattended);
    assert_eq!(rec.surface, DECISION_RECORD_SURFACE);
    assert!(matches!(
        rec.threat_class,
        DecisionThreatClass::Medium | DecisionThreatClass::High | DecisionThreatClass::Critical
    ));
    assert_eq!(rec.payload_sha256, DecisionRecord::payload_sha256(&text));
    assert!(
        rec.reason_codes
            .iter()
            .any(|c| c.contains("trust_remote_code") || c.contains("remote_code_loader")),
        "blocked record must name the signal/threat class, not the raw paste: {:?}",
        rec.reason_codes
    );
    assert!(rec.override_rationale.is_none());
    assert!(rec.prev_verdict.is_none());
}

#[test]
fn decision_record_json_round_trip() {
    for rel in [
        "benign/model_card_clean.md",
        "blocked/trust_remote_code_loader.txt",
    ] {
        let text = public_fixture(rel);
        let rec = DecisionRecord::from_unattended_ingest(&text);
        let json = rec.to_log_json().expect("serialize log");
        let back = DecisionRecord::from_log_json(&json).expect("deserialize log");
        assert_eq!(rec, back, "round-trip must preserve {rel}");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["surface"], DECISION_RECORD_SURFACE);
        assert!(
            v["verdict"] == "admit" || v["verdict"] == "block",
            "unattended fixture verdict must be admit|block: {}",
            v["verdict"]
        );
        assert_eq!(v["actor"], "unattended");
        assert!(v.get("payload").is_none());
        assert!(v.get("content").is_none());
    }
}

#[test]
fn decision_record_refuses_raw_payload_in_the_log() {
    let text = public_fixture("blocked/trust_remote_code_loader.txt");
    let rec = DecisionRecord::from_unattended_ingest(&text);
    let json = rec.to_log_json().expect("serialize log");

    assert!(
        !json.contains(&text),
        "log JSON must not contain the raw fixture body"
    );
    assert!(
        !json.contains("DEFENSIVE TEST FIXTURE"),
        "log JSON must not contain fixture header prose"
    );
    assert!(
        !json.contains("load_dataset"),
        "log JSON must not contain raw loader paste"
    );
    assert!(
        !json.contains("example/name"),
        "log JSON must not contain fixture dataset name"
    );

    let admitted = public_fixture("benign/model_card_clean.md");
    let admit_json = DecisionRecord::from_unattended_ingest(&admitted)
        .to_log_json()
        .expect("serialize admit log");
    assert!(!admit_json.contains("Image Classifier Card"));
    assert!(!admit_json.contains("everyday photos"));
    assert!(!admit_json.contains(&admitted));

    let mut sneaky: serde_json::Value = serde_json::from_str(&json).unwrap();
    sneaky["payload"] = serde_json::json!(text);
    let sneaky_json = serde_json::to_string(&sneaky).unwrap();
    let err = DecisionRecord::from_log_json(&sneaky_json).expect_err("raw payload field must refuse");
    assert!(
        matches!(err, DecisionRecordError::Refused(ref msg) if msg.contains("payload")),
        "expected Refused(payload…), got {err:?}"
    );

    sneaky = serde_json::from_str(&json).unwrap();
    sneaky["content"] = serde_json::json!(text);
    let err = DecisionRecord::from_log_json(&serde_json::to_string(&sneaky).unwrap())
        .expect_err("content field must refuse");
    assert!(matches!(err, DecisionRecordError::Refused(_)));
}

#[test]
fn human_override_records_actor_rationale_and_prev_verdict() {
    let text = public_fixture("blocked/trust_remote_code_loader.txt");
    let blocked = DecisionRecord::from_unattended_ingest(&text);
    let over = DecisionRecord::human_override(&blocked, "steward reviewed classroom demo paste")
        .expect("override with rationale");
    assert_eq!(over.verdict, DecisionVerdict::Override);
    assert_eq!(over.actor, DecisionActor::Human);
    assert_eq!(over.prev_verdict, Some(DecisionVerdict::Block));
    assert_eq!(
        over.override_rationale.as_deref(),
        Some("steward reviewed classroom demo paste")
    );
    assert_eq!(over.payload_sha256, blocked.payload_sha256);
    let json = over.to_log_json().unwrap();
    let back = DecisionRecord::from_log_json(&json).unwrap();
    assert_eq!(over, back);
    assert!(!json.contains(&text));
    // Unattended path is unchanged by the override constructor.
    assert!(IngestionScanner::admit_or_block(&text).is_err());
}

/// GE-GAP-HUMAN-OVERRIDE — public blocked ingest + companion override note.
/// Docs+test completeness only. Not a claim that override is safe.
#[test]
fn human_override_blocked_public_fixture_is_complete() {
    let ingest = public_fixture("blocked/human_override_classroom_demo.txt");
    let note = public_fixture("human-override/classroom_demo.override.md");

    assert!(
        ingest.contains("trust_remote_code"),
        "blocked override fixture must remain a pattern marker"
    );
    assert!(
        IngestionScanner::admit_or_block(&ingest).is_err(),
        "unattended Medium+ must still BLOCK; human_override is not a scanner bypass"
    );

    let blocked = DecisionRecord::from_unattended_ingest(&ingest);
    assert_eq!(blocked.verdict, DecisionVerdict::Block);
    assert_eq!(blocked.actor, DecisionActor::Unattended);
    assert!(blocked.override_rationale.is_none());
    assert!(blocked.prev_verdict.is_none());

    let empty = DecisionRecord::human_override(&blocked, "");
    assert!(
        matches!(empty, Err(DecisionRecordError::Refused(ref msg) if msg.contains("override_rationale"))),
        "empty rationale must be refused: {empty:?}"
    );
    let whitespace = DecisionRecord::human_override(&blocked, "   \n\t  ");
    assert!(
        matches!(whitespace, Err(DecisionRecordError::Refused(_))),
        "whitespace-only rationale must be refused: {whitespace:?}"
    );

    let rationale = note.trim();
    assert!(
        !rationale.is_empty(),
        "companion override note must be non-empty rationale"
    );
    assert!(
        IngestionScanner::admit_or_block(rationale).is_ok(),
        "companion override note is rationale prose, not a second blocked ingest"
    );

    let over = DecisionRecord::human_override(&blocked, rationale)
        .expect("companion note is a non-empty rationale");
    assert_eq!(over.actor, DecisionActor::Human);
    assert_eq!(over.verdict, DecisionVerdict::Override);
    assert_eq!(over.prev_verdict, Some(DecisionVerdict::Block));
    assert_eq!(over.payload_sha256, blocked.payload_sha256);
    let logged_rationale = over
        .override_rationale
        .as_deref()
        .expect("override must carry rationale");
    assert!(!logged_rationale.trim().is_empty());
    assert_eq!(logged_rationale, rationale);

    let json = over.to_log_json().expect("serialize override log");
    let back = DecisionRecord::from_log_json(&json).expect("deserialize override log");
    assert_eq!(over, back);
    assert!(
        !json.contains(&ingest),
        "to_log_json must not contain the raw blocked ingest"
    );
    assert!(
        !json.contains("DEFENSIVE TEST FIXTURE"),
        "to_log_json must not contain the ingest fixture header"
    );
    assert!(
        !json.contains("trust_remote_code"),
        "to_log_json must not contain the raw loader marker"
    );
    assert!(
        json.contains("\"verdict\":\"override\"") || json.contains("\"verdict\": \"override\""),
        "log must name verdict=override: {json}"
    );

    // Constructor does not change the living unattended gate.
    assert!(IngestionScanner::admit_or_block(&ingest).is_err());
}
