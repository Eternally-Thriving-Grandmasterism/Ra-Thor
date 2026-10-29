//! DEMO-HARNESS-1 headless integration tests (auto-answers, no network, no LLM).
//! AG-SML v1.2 | Contact: info@Rathor.ai

use lattice_conductor_v14::prefix_risk::PrefixAction;
use mercy_demo_harness::*;
use mercy_security::{DecisionActor, DecisionRecord, DecisionVerdict};
use serde_json::{json, Value};

fn trail_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("mdh-{tag}-{}.jsonl", uuid::Uuid::new_v4()))
}

fn run_scenario(tag: &str) -> (Harness, Vec<StepOutcome>) {
    let mut h = Harness::new(trail_path(tag)).unwrap();
    let mut answers = scenario_auto_answers();
    let mut outs = Vec::new();
    for (i, s) in scenario().iter().enumerate() {
        let (resp, o) = h.handle_rpc(&rpc_line(i + 1, s), &mut answers);
        let r: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(r["jsonrpc"], "2.0");
        assert_eq!(r["id"], json!(i + 1));
        outs.push(o.expect("tools/call returns an outcome"));
    }
    (h, outs)
}

fn records(h: &Harness) -> Vec<DecisionRecord> {
    std::fs::read_to_string(&h.trail.path).unwrap().lines()
        .filter(|l| !l.contains("\"source\":\"harness\""))
        .map(|l| DecisionRecord::from_log_json(l).expect("valid DecisionRecord line"))
        .collect()
}

fn harness_events(h: &Harness, ev: &str) -> Vec<Value> {
    h.trail.lines().into_iter().filter(|v| v["source"] == "harness" && v["event"] == ev).collect()
}

#[test]
fn each_scenario_step_has_its_expected_verdict() {
    let (_h, outs) = run_scenario("verdicts");
    let steps = scenario();
    assert!(steps.len() >= 8);
    for (s, o) in steps.iter().zip(&outs) {
        assert_eq!(o.demo, s.expect, "step {} ({}) {}", o.step, o.tool, s.label);
    }
    // Specific gate sources.
    assert!(matches!(outs[0].prefix_action, PrefixAction::Halt { .. }));
    assert!(outs[4].gate_error.as_deref().unwrap().starts_with("containment"));
    assert!(outs[5].gate_error.as_deref().unwrap().starts_with("containment"));
    assert!(outs[6].gate_error.as_deref().unwrap().starts_with("harm_refusal"));
    assert!(matches!(&outs[10].prefix_action, PrefixAction::Escalate { councils, .. } if councils == &vec![13]));
    assert!(outs[11].gate_error.as_deref().unwrap().starts_with("action_governor"));
    // Gap step is labelled and admits.
    assert!(outs[7].known_gap && outs[7].ran && outs[7].demo == DemoVerdict::Admit);
    // Only Admit and approved Review ran.
    let ran: Vec<usize> = outs.iter().filter(|o| o.ran).map(|o| o.step).collect();
    assert_eq!(ran, vec![2, 3, 4, 8, 10]);
}

#[test]
fn a_decision_record_and_a_harness_step_line_are_written_per_step() {
    let (h, outs) = run_scenario("records");
    let recs = records(&h);
    let steps = harness_events(&h, "step");
    assert_eq!(steps.len(), outs.len());
    for o in &outs {
        let id = o.decision_record.record_id;
        assert!(recs.iter().any(|r| r.record_id == id), "record for step {}", o.step);
        let line = steps.iter().find(|v| v["step"] == json!(o.step)).unwrap();
        assert_eq!(line["decision_record_id"], json!(id.to_string()));
        assert_eq!(line["demo_outcome"], json!(o.demo.as_str()));
    }
    // Unattended records only, except the one human Override for the approved Review.
    let overrides: Vec<_> = recs.iter().filter(|r| r.verdict == DecisionVerdict::Override).collect();
    assert_eq!(overrides.len(), 1);
}

#[test]
fn gap_step_quotes_marker_text_verbatim() {
    let (h, outs) = run_scenario("gap");
    let gap = &outs[7];
    let rendered = render_step(&scenario()[7], gap);
    assert!(rendered.contains(GAP_TEST_NAME));
    assert!(rendered.contains("GATE-EVAL-1 gap marker: tool-use function call class is not in IngestionScanner tables."));
    assert!(rendered.contains("They currently ADMIT. Do not read this as “safe to ship.”"));
    let line = harness_events(&h, "step").into_iter().find(|v| v["step"] == json!(8)).unwrap();
    assert!(line["note"].as_str().unwrap().contains(GAP_MARKER_COMMENT));
}

#[test]
fn approve_after_block_or_halt_is_refused_with_a_harness_event() {
    let (mut h, outs) = run_scenario("refuse");
    let runs_before = h.tool_runs;
    let recs_before = records(&h).len();
    for idx in [0usize, 4, 6, 11] {
        // 0 = prefix Halt, 4 = containment Block, 6 = harm refusal, 11 = governor
        let o = &outs[idx];
        assert_eq!(o.demo, DemoVerdict::Block);
        assert!(o.human.is_none(), "Block/Halt never gets a prompt");
        let r = h.attempt_approve(o, "operator tried to approve anyway");
        assert!(r.is_err());
    }
    // Tool did not run, no Override record minted.
    assert_eq!(h.tool_runs, runs_before);
    let recs = records(&h);
    assert_eq!(recs.len(), recs_before);
    assert!(!recs.iter().any(|r| r.verdict == DecisionVerdict::Override && [0usize, 4, 6, 11].iter().any(|i| outs[*i].decision_record.payload_sha256 == r.payload_sha256)));
    // approve_refused events: harness-labelled, no verdict field at all.
    let ev = harness_events(&h, "approve_refused");
    assert_eq!(ev.len(), 4);
    for e in &ev {
        assert_eq!(e["source"], "harness");
        let o = e.as_object().unwrap();
        assert!(!o.contains_key("verdict"), "no verdict field: {e}");
        assert!(!o.contains_key("demo_outcome"));
        assert!(!o.values().any(|v| v == "override" || v == "admit"));
        assert_eq!(e["rationale"], "operator tried to approve anyway");
        assert!(DecisionRecord::from_log_json(&e.to_string()).is_err(), "never parses as a DecisionRecord");
        assert!(render_trail_line(&e.to_string()).starts_with("harness event"));
    }
    // Escalate is not approvable either.
    assert!(h.attempt_approve(&outs[10], "x").is_err());
}

#[test]
fn deny_writes_a_harness_line_never_calls_human_override_and_tool_does_not_run() {
    let (h, outs) = run_scenario("deny");
    let d = &outs[8];
    assert_eq!(d.demo, DemoVerdict::Review);
    assert!(matches!(d.human, Some(Answer::Deny(_))));
    assert!(!d.ran && d.override_record.is_none());
    assert!(!h.sandbox.root.join("draft2.txt").exists());
    // No Override record carries the denied call's payload hash.
    assert!(!records(&h).iter().any(|r| r.verdict == DecisionVerdict::Override && r.payload_sha256 == d.decision_record.payload_sha256));
    let ev = harness_events(&h, "deny");
    assert_eq!(ev.len(), 1);
    let o = ev[0].as_object().unwrap();
    assert_eq!(ev[0]["source"], "harness");
    assert_eq!(ev[0]["step"], json!(9));
    assert!(!o.contains_key("verdict") && !o.contains_key("demo_outcome"));
    assert!(render_trail_line(&ev[0].to_string()).starts_with("harness event"));
}

#[test]
fn review_approve_writes_override_human_with_prev_verdict_and_runs_the_tool() {
    let (h, outs) = run_scenario("approve");
    let a = &outs[3];
    assert_eq!(a.demo, DemoVerdict::Review);
    let ovr = a.override_record.as_ref().expect("override record");
    assert_eq!(ovr.verdict, DecisionVerdict::Override);
    assert_eq!(ovr.actor, DecisionActor::Human);
    assert_eq!(ovr.prev_verdict, Some(DecisionVerdict::Admit));
    assert_eq!(ovr.override_rationale.as_deref(), Some("demo: draft write reviewed by operator"));
    assert_eq!(record_label(ovr), "Override (human), prev_verdict=admit");
    assert!(a.ran);
    assert_eq!(std::fs::read_to_string(h.sandbox.root.join("draft.txt")).unwrap(), "summary of notes");
    assert!(records(&h).iter().any(|r| r.record_id == ovr.record_id));
}

#[test]
fn rpc_shim_lists_tools_and_rejects_unknown_methods() {
    let mut h = Harness::new(trail_path("rpc")).unwrap();
    let mut a = AutoAnswers(vec![]);
    let (r, _) = h.handle_rpc(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#, &mut a);
    let v: Value = serde_json::from_str(&r).unwrap();
    assert_eq!(v["result"]["tools"].as_array().unwrap().len(), 5);
    let (r, _) = h.handle_rpc(r#"{"jsonrpc":"2.0","id":2,"method":"nope"}"#, &mut a);
    assert_eq!(serde_json::from_str::<Value>(&r).unwrap()["error"]["code"], -32601);
    let (r, _) = h.handle_rpc("not json", &mut a);
    assert_eq!(serde_json::from_str::<Value>(&r).unwrap()["error"]["code"], -32700);
}

#[test]
fn sandbox_tools_stay_offline_and_confined() {
    let h = Harness::new(trail_path("sb")).unwrap();
    assert!(h.sandbox.run("read_file", &json!({"path":"../etc/passwd"})).is_err());
    assert!(h.sandbox.run("read_file", &json!({"path":"/etc/passwd"})).is_err());
    assert!(h.sandbox.run("run_cmd", &json!({"cmd":"ls /"})).is_err());
    assert_eq!(h.sandbox.run("run_cmd", &json!({"cmd":"echo hi there"})).unwrap(), "hi there");
    assert!(h.sandbox.run("fake_http", &json!({"url":"https://example.com"})).unwrap().contains("no network used"));
}
