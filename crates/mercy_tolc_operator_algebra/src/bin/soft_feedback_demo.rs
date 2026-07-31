//! soft_feedback_demo.rs
//!
//! Public dual-repo soft feedback demonstration (v0.5.5).
//! Exercises SoftFeedbackBridge under concurrent multi-zone stress and prints
//! sealed SoftFeedbackEvent + ZoneSnapshot payloads (the Powrush-MMO contract).
//!
//! Run:
//!   cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo
//!   cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --agents 12000 --zones 4
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi | info@Rathor.ai | Thunder locked. Yoi ⚡

use mercy_tolc_operator_algebra::{SoftFeedbackBridge, Valence, AMBIENT_DIM, MERCY_DIM};
use std::env;

fn parse_args() -> (usize, usize) {
    let args: Vec<String> = env::args().collect();
    let mut agents = 12_000usize;
    let mut zones = 4usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--agents" if i + 1 < args.len() => {
                agents = args[i + 1].parse().unwrap_or(agents);
                i += 2;
            }
            "--zones" if i + 1 < args.len() => {
                zones = args[i + 1].parse().unwrap_or(zones);
                i += 2;
            }
            _ => i += 1,
        }
    }
    (agents, zones.max(1))
}

fn main() {
    let (n_agents, n_zones) = parse_args();

    println!("══════════════════════════════════════════════════════════════");
    println!("  Ra-Thor · Soft Feedback Dual-Repo Demo");
    println!("  Ambient ℝ^{AMBIENT_DIM} ⊃ Mercy ℝ^{MERCY_DIM} · Sealed protocol → Powrush-MMO");
    println!("  Contact: info@Rathor.ai");
    println!("══════════════════════════════════════════════════════════════\n");
    println!("  Agents: {n_agents}   Zones: {n_zones}\n");

    let mut bridge = SoftFeedbackBridge::new(n_zones);
    bridge.lattice.purify_period = 1_000;

    let mut sample_events = Vec::new();

    for i in 0..n_agents {
        let zone = i % n_zones;
        let valence = match i % 3 {
            0 => Valence::HIGH,
            1 => Valence::MID,
            _ => Valence::new(0.05),
        };
        let energy = 0.8 + (zone as f64) * 0.35 + ((i % 7) as f64) * 0.05;
        let ev = bridge.ingest_scalar_grief(zone, energy, valence);
        if sample_events.len() < 6 && (i % (n_agents / 6).max(1) == 0) {
            sample_events.push(ev);
        }
    }

    let rhos = bridge.global_purify();
    let max_rho = rhos.iter().cloned().fold(0.0_f64, f64::max);

    let drained = bridge.drain_events();
    let snaps = bridge.snapshots();

    println!("──────────────────────────────────────────────────────────────");
    println!("  Sample sealed SoftFeedbackEvent payloads (dual-repo contract)");
    println!("──────────────────────────────────────────────────────────────");
    for ev in &sample_events {
        println!(
            "  {{ zone_id: {}, grief_load: {:.4}, valence: {:.3}, under_floor: {}, tick: {} }}",
            ev.zone_id, ev.grief_load, ev.valence, ev.under_floor, ev.tick
        );
    }

    println!("\n──────────────────────────────────────────────────────────────");
    println!("  ZoneSnapshot telemetry");
    println!("──────────────────────────────────────────────────────────────");
    for s in &snaps {
        println!(
            "  Zone {}: grief_absorbed={:>10.3}  vectors={:>6}  ρ={:.3e}",
            s.zone_id, s.grief_absorbed, s.vectors_processed, s.last_rho
        );
    }

    println!("\n──────────────────────────────────────────────────────────────");
    println!("  Summary");
    println!("──────────────────────────────────────────────────────────────");
    println!("  Events recorded (drained): {}", drained.len());
    println!(
        "  Total grief absorbed:      {:.3}",
        snaps.iter().map(|s| s.grief_absorbed).sum::<f64>()
    );
    println!("  Final max zone ρ:          {:.3e}", max_rho);

    let high_soft = sample_events
        .iter()
        .any(|e| e.valence > 0.99 && e.grief_load < 1e-4);
    let low_hard = sample_events
        .iter()
        .any(|e| e.valence < 0.1 && e.grief_load > 0.5);
    let zones_ok = snaps.len() == n_zones && snaps.iter().all(|s| s.vectors_processed > 0);
    let purity_ok = max_rho < 1e-9;

    println!("\n  Verification");
    println!(
        "    High-valence soft path:     {}",
        if high_soft { "PASS" } else { "FAIL" }
    );
    println!(
        "    Low-valence exposure:       {}",
        if low_hard { "PASS" } else { "FAIL" }
    );
    println!(
        "    All zones active:           {}",
        if zones_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "    Basis purity (max ρ):       {}",
        if purity_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "    Event drain non-empty:      {}",
        if !drained.is_empty() {
            "PASS"
        } else {
            "FAIL"
        }
    );

    if high_soft && low_hard && zones_ok && purity_ok && !drained.is_empty() {
        println!("\n  ★  ALL GATES PASSED — soft feedback dual-repo protocol is live.");
        println!("     Powrush RaThorBridge::report_zone_grief mirrors this event shape.");
    } else {
        println!("\n  ⚠  One or more gates failed.");
    }
    println!("\n  Thunder locked. Yoi ⚡\n");
}
