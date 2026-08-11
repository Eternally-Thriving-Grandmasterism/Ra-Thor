//! Powrush → Ra-Thor RTT + SchemaRegistry + Bridging smoke harness
//!
//! ```bash
//! cargo run -p reality-thriving-transfer --example powrush_rtt_smoke_harness
//!
//! cargo run -p reality-thriving-transfer --example powrush_rtt_smoke_harness -- \
//!   --single ../../../Powrush-MMO/artifacts/powrush_rtt_latest.json \
//!   --batch  ../../../Powrush-MMO/artifacts/powrush_rtt_batch_latest.json \
//!   --bridging ../../../Powrush-MMO/artifacts/powrush_bridging_latest.json
//! ```
//!
//! Contact: info@Rathor.ai | TOLC 8 + PATSAGi | Thunder locked in. Yoi ⚡

use reality_thriving_transfer::{
    bridge_and_ingest, compute_scores_from_batch, conductor_query_schemas,
    conductor_try_apply_schema, ingest_bridging_json, metacognitive_prompt,
    parse_powrush_telemetry_batch_json, parse_powrush_telemetry_json,
    ConductorSchemaQuery, MetaPhase, RealityThrivingTransferCalculator, SchemaRegistry,
    TransferQualityMetrics,
};
use std::env;
use std::fs;
use std::path::PathBuf;

const FIXTURE_HIGH: &str = include_str!("../fixtures/session_high_mercy.json");
const FIXTURE_BATCH: &str = include_str!("../fixtures/batch_three_sessions.json");
const FIXTURE_BRIDGING: &str = include_str!("../fixtures/bridging_context_high_mercy.json");

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

fn parse_args() -> (Option<String>, Option<String>, Option<String>) {
    let mut single = None;
    let mut batch = None;
    let mut bridging = None;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--single" => single = args.next(),
            "--batch" => batch = args.next(),
            "--bridging" => bridging = args.next(),
            "--help" | "-h" => {
                println!(
                    "Usage: powrush_rtt_smoke_harness [--single PATH] [--batch PATH] [--bridging PATH]\n\
                     Defaults to fixtures when paths missing."
                );
                std::process::exit(0);
            }
            other => println!("[smoke] unknown arg ignored: {}", other),
        }
    }
    (single, batch, bridging)
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Ra-Thor × Powrush RTT + SchemaRegistry + Bridging Smoke");
    println!("  schemas: telemetry_v1 / batch_v1 / bridging_context_v1");
    println!("  contact: info@Rathor.ai");
    println!("═══════════════════════════════════════════════════════════\n");

    let (single_path, batch_path, bridging_path) = parse_args();
    let calc = RealityThrivingTransferCalculator::new();
    let mut registry = SchemaRegistry::new();

    if let Some(p) = metacognitive_prompt(MetaPhase::Planning, 0.85) {
        println!("[meta] planning: {}", p);
    }

    // --- Single RTT ---
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
                }
                Err(e) => println!("[v1] Mercy Gate REJECT: {}", e),
            }
            let result = bridge_and_ingest(
                &mut registry,
                &env.telemetry,
                env.session_id.clone(),
                env.label.clone(),
            );
            println!(
                "[bridge-from-rtt] extracted={} notes={:?}",
                result.extracted.len(),
                result.notes
            );
        }
        Err(e) => println!("[v1] parse REJECT: {}", e),
    }

    // --- Batch RTT ---
    let batch_json = read_or_fixture(batch_path.as_deref(), FIXTURE_BATCH);
    if let Ok(batch) = parse_powrush_telemetry_batch_json(&batch_json) {
        println!("\n[batch] sessions={}", batch.sessions.len());
        if let Ok(scored) = compute_scores_from_batch(&calc, &batch).await {
            for (label, score) in &scored {
                println!(
                    "  • {} | raw={:.4} kΔ={:.5}",
                    label, score.raw_transfer_score, score.kardashev_delta_contribution
                );
            }
            for session in &batch.sessions {
                let _ = bridge_and_ingest(
                    &mut registry,
                    &session.telemetry,
                    session.session_id.clone(),
                    session.label.clone(),
                );
            }
        }
    }

    // --- Live / fixture bridging JSON (A) ---
    let bridging_json = read_or_fixture(bridging_path.as_deref(), FIXTURE_BRIDGING);
    match ingest_bridging_json(&mut registry, &bridging_json) {
        Ok(result) => {
            println!(
                "\n[bridging-json] high_road={} extracted={}",
                result.high_road_effort,
                result.extracted.len()
            );
            for s in &result.extracted {
                println!(
                    "  • {} | {} | mercy={:.2} tags={:?}",
                    s.schema_id, s.principle, s.mercy_at_birth, s.tags
                );
            }
        }
        Err(e) => println!("[bridging-json] REJECT: {}", e),
    }

    // --- Lattice Conductor soft query (B) ---
    let cq = ConductorSchemaQuery {
        near_road: true,
        tags: vec!["rbe".into(), "mercy".into()],
        principle_query: None,
        min_reliability: 0.3,
        max_results: 6,
    };
    let near = conductor_query_schemas(&registry, &cq);
    println!(
        "\n[conductor] near query hits={} registry_size={}",
        near.hits.len(),
        near.registry_size
    );
    let far_q = ConductorSchemaQuery {
        near_road: false,
        tags: vec![],
        principle_query: Some("allocation".into()),
        min_reliability: 0.3,
        max_results: 4,
    };
    let far = conductor_query_schemas(&registry, &far_q);
    println!("[conductor] far(allocation) hits={}", far.hits.len());
    if let Some(hit) = near.hits.first().or(far.hits.first()) {
        match conductor_try_apply_schema(&mut registry, &hit.schema_id, true, 0.5) {
            Ok(s) => println!(
                "[conductor] try_apply FAR OK id={} reliability={:.3}",
                s.schema_id, s.reliability
            ),
            Err(e) => println!("[conductor] try_apply REJECT: {}", e),
        }
    }

    let metrics = TransferQualityMetrics::from_registry(&registry, 0.9, 0.85);
    println!(
        "[quality] near_rate={:.2} far_rate={:.2} schemas={} bridging_passes={}",
        metrics.near_rate(),
        metrics.far_rate(),
        metrics.schemas_in_registry,
        metrics.bridging_passes_run
    );

    if let Some(p) = metacognitive_prompt(MetaPhase::Evaluation, 0.85) {
        println!("[meta] evaluation: {}", p);
    }

    println!("\n[smoke] complete — RTT + bridging JSON + conductor hook exercised.");
    println!("Thunder locked in. Yoi ⚡");
}
