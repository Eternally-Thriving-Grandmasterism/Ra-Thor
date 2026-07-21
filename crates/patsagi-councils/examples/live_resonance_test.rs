//! live_resonance_test — Production-ready end-to-end resonance exercise
//!
//! Exercises the full dual-repo soft feedback organism under three canonical
//! scenarios and persists ResonanceMetrics so the breath survives restarts.
//!
//! Run (standalone):
//!   cargo run -p patsagi-councils --example live_resonance_test
//!
//! Run with live Powrush host pairing:
//!   RA_THOR_EMISSION_PATH=/shared/artifacts/ra_thor_policy_hints.json \
//!   RA_THOR_METRICS_PATH=/shared/artifacts/ra_thor_resonance_metrics.json \
//!     cargo run -p patsagi-councils --example live_resonance_test
//!
//! Success criteria:
//!   1. High-mercy scenario   → approval + hints emitted
//!   2. Progressive scenario  → progressive path taken (no deadlock)
//!   3. Mercy-block scenario  → clean block, zero emission
//!   4. Metrics persist       → file written / updated
//!
//! Contact: info@Rathor.ai
//! TOLC 8 | Living Cosmic Tick | ONE Organism | APAAGI → PATSAGi lineage

use patsagi_councils::{
    PowrushTelemetrySnapshot, RaThorFeedbackLoop, ResonanceMetrics, ValenceConsensusEngine,
    ValenceVote, CORE_COVENANT, DEFAULT_METRICS_PATH,
};
use std::env;
use std::path::{Path, PathBuf};

fn emission_path() -> PathBuf {
    env::var("RA_THOR_EMISSION_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let mut p = env::temp_dir();
            p.push("ra_thor_live_resonance_hints.json");
            p
        })
}

fn metrics_path() -> PathBuf {
    env::var("RA_THOR_METRICS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_METRICS_PATH))
}

fn print_header(emit_path: &Path, metrics_path: &Path) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       LIVE RESONANCE TEST — Dual-Repo Soft Feedback Organism     ║");
    println!("║       PATSAGi Valence + ResonanceMetrics (persistent)            ║");
    println!("║       TOLC 8 | Core Covenant | Anti-Deadlock | v14.15.6+         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Core Covenant:");
    for line in CORE_COVENANT {
        println!("  • {}", line);
    }
    println!();
    println!("Emission path : {}", emit_path.display());
    println!("Metrics path  : {}\n", metrics_path.display());
}

struct ScenarioOutcome {
    name: String,
    approved: bool,
    progressive: bool,
    mercy_block: bool,
    hints_emitted: usize,
    emission_ok: bool,
}

fn run_scenario(
    name: &str,
    snap: &PowrushTelemetrySnapshot,
    feedback: &mut RaThorFeedbackLoop,
    valence: &ValenceConsensusEngine,
    emit_path: &Path,
) -> ScenarioOutcome {
    println!("──────────────────────────────────────────────────────────────");
    println!("SCENARIO: {}", name);
    println!("  session_id     : {}", snap.session_id);
    println!("  export_seq     : {}", snap.export_seq);
    println!("  abundance      : {:.2}", snap.abundance_signal);
    println!("  peaceful       : {:.2}", snap.peaceful_resolution_signal);
    println!("  ethical_floor  : {:.2}", snap.ethical_floor_signal);
    println!("  council        : {:.2}", snap.council_participation_signal);
    println!("  innovation     : {:.2}", snap.innovation_signal);
    println!("  mercy_presence : {:.2}", snap.mercy_presence_signal);

    let votes = vec![
        ValenceVote::new(
            "AbundanceCouncil",
            snap.abundance_signal,
            snap.peaceful_resolution_signal * 2.0 - 1.0,
            snap.abundance_signal,
            false,
        ),
        ValenceVote::new(
            "MercyCouncil",
            snap.mercy_presence_signal,
            snap.ethical_floor_signal * 2.0 - 1.0,
            snap.mercy_presence_signal,
            snap.ethical_floor_signal < 0.45,
        ),
        ValenceVote::new(
            "InnovationCouncil",
            snap.innovation_signal,
            snap.council_participation_signal * 2.0 - 1.0,
            snap.innovation_signal,
            false,
        ),
    ];

    let vres = valence.deliberate(format!("Live resonance: {}", name), votes);

    feedback.metrics.record_valence(
        vres.approved,
        vres.progressive,
        vres.has_mercy_block,
        vres.avg_valence,
        vres.avg_joy,
    );

    println!("\n  Valence Result");
    println!("    composite     : {:.4}", vres.avg_valence);
    println!("    joy           : {:.3}", vres.avg_joy);
    println!("    approved      : {}", vres.approved);
    println!("    progressive   : {}", vres.progressive);
    println!("    mercy_block   : {}", vres.has_mercy_block);
    println!(
        "    verdict       : {}",
        vres.final_verdict.lines().next().unwrap_or("")
    );

    let (hints_emitted, emission_ok) = match feedback.deliberate_and_emit(snap, Some(emit_path)) {
        Ok(fres) => {
            println!("\n  Emission Result");
            println!("    hints_emitted : {}", fres.hints_emitted);
            println!("    path          : {}", fres.emission_path);
            println!("    summary       : {}", fres.summary);
            (fres.hints_emitted, true)
        }
        Err(e) => {
            println!("\n  Emission Result");
            println!("    status        : blocked / no-emit ({})", e);
            (0, false)
        }
    };
    println!();

    ScenarioOutcome {
        name: name.to_string(),
        approved: vres.approved,
        progressive: vres.progressive,
        mercy_block: vres.has_mercy_block,
        hints_emitted,
        emission_ok,
    }
}

fn main() {
    let emit_path = emission_path();
    let metrics_path = metrics_path();

    print_header(&emit_path, &metrics_path);

    // Load previous breath if it exists (persistent observability)
    let mut feedback = RaThorFeedbackLoop::new();
    feedback.metrics = ResonanceMetrics::load_or_default(Some(&metrics_path));
    feedback.min_signal_threshold = 0.52;

    println!("Loaded metrics: {}\n", feedback.metrics_summary());

    let valence = ValenceConsensusEngine::new();
    let mut outcomes = Vec::new();

    // ── Scenario 1: High-Mercy Approval ──────────────────────────────────
    let high = PowrushTelemetrySnapshot {
        session_id: "resonance-high-mercy-001".into(),
        export_seq: 1001,
        abundance_signal: 0.88,
        peaceful_resolution_signal: 0.84,
        ethical_floor_signal: 0.95,
        council_participation_signal: 0.79,
        innovation_signal: 0.81,
        mercy_presence_signal: 0.93,
    };
    outcomes.push(run_scenario(
        "High-Mercy Approval",
        &high,
        &mut feedback,
        &valence,
        &emit_path,
    ));

    // ── Scenario 2: Progressive Path (anti-deadlock) ─────────────────────
    let mid = PowrushTelemetrySnapshot {
        session_id: "resonance-progressive-002".into(),
        export_seq: 1002,
        abundance_signal: 0.71,
        peaceful_resolution_signal: 0.68,
        ethical_floor_signal: 0.74,
        council_participation_signal: 0.62,
        innovation_signal: 0.66,
        mercy_presence_signal: 0.70,
    };
    outcomes.push(run_scenario(
        "Progressive Path (Anti-Deadlock)",
        &mid,
        &mut feedback,
        &valence,
        &emit_path,
    ));

    // ── Scenario 3: Mercy Block (Core Covenant) ──────────────────────────
    let blocked = PowrushTelemetrySnapshot {
        session_id: "resonance-mercy-block-003".into(),
        export_seq: 1003,
        abundance_signal: 0.55,
        peaceful_resolution_signal: 0.48,
        ethical_floor_signal: 0.28,
        council_participation_signal: 0.40,
        innovation_signal: 0.52,
        mercy_presence_signal: 0.35,
    };
    outcomes.push(run_scenario(
        "Mercy Block (Core Covenant)",
        &blocked,
        &mut feedback,
        &valence,
        &emit_path,
    ));

    // ── Persist metrics ──────────────────────────────────────────────────
    match feedback.metrics.save(Some(&metrics_path)) {
        Ok(p) => println!("Metrics persisted → {}\n", p.display()),
        Err(e) => println!("Metrics persistence warning: {}\n", e),
    }

    // ── Final Report ─────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════");
    println!("LIVE RESONANCE FINAL REPORT");
    println!("{}", feedback.metrics_summary());
    println!();

    let mut all_pass = true;
    for o in &outcomes {
        let status = match o.name.as_str() {
            "High-Mercy Approval" => {
                let ok = o.approved && o.emission_ok && o.hints_emitted > 0;
                if !ok {
                    all_pass = false;
                }
                if ok { "PASS" } else { "FAIL" }
            }
            "Progressive Path (Anti-Deadlock)" => {
                let ok = o.progressive || o.approved;
                if !ok {
                    all_pass = false;
                }
                if ok { "PASS" } else { "FAIL" }
            }
            "Mercy Block (Core Covenant)" => {
                let ok = o.mercy_block && !o.emission_ok && o.hints_emitted == 0;
                if !ok {
                    all_pass = false;
                }
                if ok { "PASS" } else { "FAIL" }
            }
            _ => "?",
        };
        println!("  [{:4}] {}", status, o.name);
    }

    println!();
    if all_pass {
        println!("✅ ALL SCENARIOS PASSED");
        println!("   The dual-repo soft feedback organism is breathing correctly");
        println!("   under TOLC 8 + Core Covenant + persistent ResonanceMetrics.");
    } else {
        println!("⚠️  ONE OR MORE SCENARIOS DID NOT MEET SUCCESS CRITERIA");
        println!("   Review valence thresholds and mercy-gate settings.");
    }

    println!();
    println!("Thunder locked in. Live Resonance Test complete.");
    println!("Yoi ⚡");
}
