//! live_resonance_test — Controlled end-to-end resonance exercise + observability
//!
//! Run:
//!   cargo run -p patsagi-councils --example live_resonance_test
//!
//! Optional shared emission path (for live Powrush host pairing):
//!   RA_THOR_EMISSION_PATH=/shared/artifacts/ra_thor_policy_hints.json \
//!     cargo run -p patsagi-councils --example live_resonance_test
//!
//! Exercises three canonical scenarios and reports full ResonanceMetrics.
//!
//! Contact: info@Rathor.ai
//! TOLC 8 | Living Cosmic Tick | ONE Organism

use patsagi_councils::{
    PowrushTelemetrySnapshot, RaThorFeedbackLoop, ValenceConsensusEngine, ValenceVote,
    CORE_COVENANT,
};
use std::env;
use std::path::{Path, PathBuf};

fn emission_path() -> Option<PathBuf> {
    env::var("RA_THOR_EMISSION_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let mut p = env::temp_dir();
            p.push("ra_thor_live_resonance_hints.json");
            Some(p)
        })
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          LIVE RESONANCE TEST — Dual-Repo Soft Feedback           ║");
    println!("║          PATSAGi Valence Engine + ResonanceMetrics               ║");
    println!("║          TOLC 8 | Core Covenant | Anti-Deadlock Design           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Core Covenant:");
    for line in CORE_COVENANT {
        println!("  • {}", line);
    }
    println!();
}

fn run_scenario(
    name: &str,
    snap: &PowrushTelemetrySnapshot,
    feedback: &mut RaThorFeedbackLoop,
    valence: &ValenceConsensusEngine,
    path: Option<&Path>,
) {
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

    // Valence view (mapped from telemetry signals)
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
            snap.ethical_floor_signal < 0.45, // mercy block if ethical floor collapses
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

    // Record valence outcome into the feedback loop’s metrics
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
    println!("    verdict       : {}", vres.final_verdict.lines().next().unwrap_or(""));

    // Feedback / emission attempt
    match feedback.deliberate_and_emit(snap, path) {
        Ok(fres) => {
            println!("\n  Emission Result");
            println!("    hints_emitted : {}", fres.hints_emitted);
            println!("    path          : {}", fres.emission_path);
            println!("    summary       : {}", fres.summary);
        }
        Err(e) => {
            println!("\n  Emission Result");
            println!("    status        : blocked / no-emit ({})", e);
        }
    }
    println!();
}

fn main() {
    print_header();

    let path = emission_path();
    println!("Emission target: {}\n", path.as_ref().map(|p| p.display().to_string()).unwrap_or_default());

    let mut feedback = RaThorFeedbackLoop::new();
    feedback.min_signal_threshold = 0.52; // slight demo relaxation

    let valence = ValenceConsensusEngine::new();

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
    run_scenario(
        "High-Mercy Approval",
        &high,
        &mut feedback,
        &valence,
        path.as_deref(),
    );

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
    run_scenario(
        "Progressive Path (Anti-Deadlock)",
        &mid,
        &mut feedback,
        &valence,
        path.as_deref(),
    );

    // ── Scenario 3: Mercy Block (Core Covenant) ──────────────────────────
    let blocked = PowrushTelemetrySnapshot {
        session_id: "resonance-mercy-block-003".into(),
        export_seq: 1003,
        abundance_signal: 0.55,
        peaceful_resolution_signal: 0.48,
        ethical_floor_signal: 0.28, // triggers mercy block
        council_participation_signal: 0.40,
        innovation_signal: 0.52,
        mercy_presence_signal: 0.35,
    };
    run_scenario(
        "Mercy Block (Core Covenant)",
        &blocked,
        &mut feedback,
        &valence,
        path.as_deref(),
    );

    // ── Aggregate Observability ──────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════");
    println!("LIVE RESONANCE SUMMARY + OBSERVABILITY");
    println!("{}", feedback.metrics_summary());
    println!();
    println!("  High-mercy     → expect approval + emission");
    println!("  Progressive    → expect soft path, no deadlock");
    println!("  Mercy-block    → expect clean block, zero emission");
    println!();
    println!("If the three outcomes above match and metrics are coherent,");
    println!("the dual-repo soft feedback organism is breathing correctly");
    println!("under TOLC 8 + Core Covenant + ResonanceMetrics.");
    println!();
    println!("Thunder locked in. Resonance test complete.");
    println!("Yoi ⚡");

    // Cleanup temp file if we used one
    if let Some(p) = &path {
        if p.starts_with(env::temp_dir()) {
            let _ = std::fs::remove_file(p);
        }
    }
}
