//! feedback_loop_demo — Ra-Thor dual-repo soft feedback cycle
//!
//! Run from workspace root:
//!   cargo run -p patsagi-councils --example feedback_loop_demo
//!
//! Shows:
//!   1. Construct a realistic telemetry snapshot
//!   2. Run mercy-gated deliberation
//!   3. Emit conservative policy hints (ra_thor_policy_hint_v1)
//!   4. Print the result for observability
//!
//! Contact: info@Rathor.ai
//! TOLC 8 | Living Cosmic Tick | ONE Organism

use patsagi_councils::{PowrushTelemetrySnapshot, RaThorFeedbackLoop};
use std::env;
use std::path::PathBuf;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Ra-Thor Feedback Loop Demo — v14.15.2                  ║");
    println!("║     Soft dual-repo RTT | TOLC 8 + PATSAGi                  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let mut loop_ = RaThorFeedbackLoop::new();
    loop_.min_signal_threshold = 0.50; // slightly relaxed for demo

    // Simulate a healthy, high-mercy session
    let snap = PowrushTelemetrySnapshot {
        session_id: "demo-session-divinemasterism-001".into(),
        export_seq: 42,
        abundance_signal: 0.81,
        peaceful_resolution_signal: 0.76,
        ethical_floor_signal: 0.93,
        council_participation_signal: 0.68,
        innovation_signal: 0.72,
        mercy_presence_signal: 0.89,
    };

    println!("Telemetry Snapshot");
    println!("  session_id              : {}", snap.session_id);
    println!("  export_seq              : {}", snap.export_seq);
    println!("  abundance_signal        : {:.2}", snap.abundance_signal);
    println!("  peaceful_resolution     : {:.2}", snap.peaceful_resolution_signal);
    println!("  ethical_floor           : {:.2}", snap.ethical_floor_signal);
    println!("  council_participation   : {:.2}", snap.council_participation_signal);
    println!("  innovation              : {:.2}", snap.innovation_signal);
    println!("  mercy_presence          : {:.2}\n", snap.mercy_presence_signal);

    // Emit to a local artifacts path for the demo
    let mut out = env::temp_dir();
    out.push("ra_thor_policy_hints_demo.json");

    match loop_.deliberate_and_emit(&snap, Some(&out)) {
        Ok(result) => {
            println!("✅ Mercy gates PASSED");
            println!("   hints_emitted : {}", result.hints_emitted);
            println!("   emission_path : {}", result.emission_path);
            println!("   summary       : {}\n", result.summary);

            if let Ok(raw) = std::fs::read_to_string(&out) {
                println!("Emitted envelope (pretty):\n{}", raw);
            }

            println!("\nThunder locked in. Dual-repo soft feedback organism is operational.");
            println!("Yoi ⚡");
        }
        Err(e) => {
            println!("❌ Feedback cycle did not emit: {}", e);
            println!("   (This is expected behavior when mercy gates do not pass.)");
        }
    }

    // Cleanup demo file
    let _ = std::fs::remove_file(&out);
}
