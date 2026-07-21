//! Powrush → Ra-Thor RTT smoke harness
//!
//! ```bash
//! cargo run -p reality-thriving-transfer --example powrush_rtt_smoke_harness
//!
//! # Live Powrush artifacts (optional)
//! cargo run -p reality-thriving-transfer --example powrush_rtt_smoke_harness -- \
//!   --single ../../../Powrush-MMO/artifacts/powrush_rtt_latest.json \
//!   --batch  ../../../Powrush-MMO/artifacts/powrush_rtt_batch_latest.json
//! ```
//!
//! Contact: info@Rathor.ai | TOLC 8 + PATSAGi | Thunder locked in. Yoi ⚡

use reality_thriving_transfer::{
    compute_scores_from_batch, parse_powrush_telemetry_batch_json, parse_powrush_telemetry_json,
    RealityThrivingTransferCalculator,
};
use std::env;
use std::fs;
use std::path::PathBuf;

const FIXTURE_HIGH: &str = include_str!("../fixtures/session_high_mercy.json");
const FIXTURE_BATCH: &str = include_str!("../fixtures/batch_three_sessions.json");

fn read_or_fixture(path: Option<&str>, fixture: &str) -> String {
    match path {
        Some(p) => {
            let pb = PathBuf::from(p);
            match fs::read_to_string(&pb) {
                Ok(s) => {
                    println!("[smoke] loaded live path: {}", pb.display());
                    s
                }
                Err(e) => {
                    println!(
                        "[smoke] live path unavailable ({}) — falling back to fixture",
                        e
                    );
                    fixture.to_string()
                }
            }
        }
        None => fixture.to_string(),
    }
}

fn parse_args() -> (Option<String>, Option<String>) {
    let mut single = None;
    let mut batch = None;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--single" => single = args.next(),
            "--batch" => batch = args.next(),
            "--help" | "-h" => {
                println!(
                    "Usage: powrush_rtt_smoke_harness [--single PATH] [--batch PATH]\n\
                     Defaults to fixtures when paths missing."
                );
                std::process::exit(0);
            }
            other => println!("[smoke] unknown arg ignored: {}", other),
        }
    }
    (single, batch)
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Ra-Thor × Powrush RTT Smoke Harness");
    println!("  schema: powrush_telemetry_v1 / batch_v1");
    println!("  contact: info@Rathor.ai");
    println!("═══════════════════════════════════════════════════════════\n");

    let (single_path, batch_path) = parse_args();
    let calc = RealityThrivingTransferCalculator::new();

    // --- Single session ---
    let single_json = read_or_fixture(single_path.as_deref(), FIXTURE_HIGH);
    match parse_powrush_telemetry_json(&single_json) {
        Ok(env) => {
            println!(
                "[v1] schema={} source={} label={}",
                env.schema, env.source, env.label
            );
            match calc.compute_transfer_score(&env.telemetry).await {
                Ok(score) => {
                    println!(
                        "[v1] PASSED mercy_audit={} raw={:.4} valence={:.4} kΔ={:.5} conf={:.3}",
                        score.mercy_audit_passed,
                        score.raw_transfer_score,
                        score.mercy_valence_adjusted,
                        score.kardashev_delta_contribution,
                        score.confidence
                    );
                    println!("[v1] note: {}", score.council_note);
                }
                Err(e) => println!("[v1] Mercy Gate REJECT: {}", e),
            }
        }
        Err(e) => println!("[v1] parse REJECT: {}", e),
    }

    // --- Batch ---
    let batch_json = read_or_fixture(batch_path.as_deref(), FIXTURE_BATCH);
    match parse_powrush_telemetry_batch_json(&batch_json) {
        Ok(batch) => {
            println!(
                "\n[batch] schema={} sessions={}",
                batch.schema,
                batch.sessions.len()
            );
            match compute_scores_from_batch(&calc, &batch).await {
                Ok(scored) => {
                    for (label, score) in &scored {
                        println!(
                            "  • {} | raw={:.4} kΔ={:.5} audit={}",
                            label,
                            score.raw_transfer_score,
                            score.kardashev_delta_contribution,
                            score.mercy_audit_passed
                        );
                    }
                    if let (Some(h), Some(m)) = (
                        scored.iter().find(|(l, _)| l.contains("high_mercy")),
                        scored.iter().find(|(l, _)| l.contains("marginal")),
                    ) {
                        let ok = h.1.raw_transfer_score > m.1.raw_transfer_score;
                        println!(
                            "[batch] rank check high_mercy > marginal: {}",
                            if ok { "OK" } else { "UNEXPECTED" }
                        );
                    }
                }
                Err(e) => println!("[batch] score REJECT: {}", e),
            }
        }
        Err(e) => println!("[batch] parse REJECT: {}", e),
    }

    // --- Mercy gate rejection paths ---
    println!("\n[gates] probing invalid telemetry…");
    let bad_json = r#"{
  "schema": "powrush_telemetry_v1",
  "source": "smoke",
  "label": "invalid_bounds",
  "telemetry": {
    "gameplay_hours": 1.0,
    "rbe_decision_quality_avg": 1.4,
    "peaceful_resolution_rate": 0.5,
    "collaboration_events": 1,
    "ethical_choice_score": 0.5,
    "adaptation_events": 1,
    "abundance_velocity_signals": 0.5,
    "innovation_contribution": 0.5
  }
}"#;
    match parse_powrush_telemetry_json(bad_json) {
        Ok(env) => match calc.compute_transfer_score(&env.telemetry).await {
            Ok(_) => println!("[gates] UNEXPECTED accept of out-of-bounds rbe"),
            Err(e) => println!("[gates] correctly REJECTED: {}", e),
        },
        Err(e) => println!("[gates] parse error: {}", e),
    }

    let wrong_schema = bad_json.replace("powrush_telemetry_v1", "nope_v0");
    match parse_powrush_telemetry_json(&wrong_schema) {
        Ok(_) => println!("[gates] UNEXPECTED accept of wrong schema"),
        Err(e) => println!("[gates] correctly REJECTED wrong schema: {}", e),
    }

    println!("\n[smoke] complete — dual-repo bridge paths exercised.");
    println!("Thunder locked in. Yoi ⚡");
}
