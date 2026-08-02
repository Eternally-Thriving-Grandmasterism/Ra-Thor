//! soft_feedback_demo.rs
//!
//! Public dual-repo soft feedback demonstration (v0.5.14).
//! Optional `--json` emits LatticeHealthReport + sample events (machine-readable).
//! CI gate: healthy && health_score ≥ 0.5
//!
//! Run:
//!   cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo
//!   cargo run -p mercy_tolc_operator_algebra --bin soft_feedback_demo -- --agents 12000 --zones 4 --json
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi | info@Rathor.ai | Thunder locked. Yoi ⚡

use mercy_tolc_operator_algebra::{SoftFeedbackBridge, SoftFeedbackEvent, Valence, AMBIENT_DIM, MERCY_DIM};
use std::env;

fn parse_args() -> (usize, usize, bool) {
    let args: Vec<String> = env::args().collect();
    let mut agents = 12_000usize;
    let mut zones = 4usize;
    let mut json = false;
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
            "--json" => {
                json = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    (agents, zones.max(1), json)
}

fn main() {
    let (n_agents, n_zones, json_mode) = parse_args();

    let mut bridge = SoftFeedbackBridge::new(n_zones);
    bridge.lattice.purify_period = 1_000;

    let mut sample_events: Vec<SoftFeedbackEvent> = Vec::new();

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

    let _rhos = bridge.global_purify();
    let health = bridge.health_report();
    let drained = bridge.drain_events();

    if json_mode {
        #[derive(serde::Serialize)]
        struct DemoExport<'a> {
            health: &'a mercy_tolc_operator_algebra::LatticeHealthReport,
            sample_events: &'a [SoftFeedbackEvent],
            events_drained: usize,
            gates: DemoGates,
        }
        #[derive(serde::Serialize)]
        struct DemoGates {
            high_valence_soft: bool,
            low_valence_exposure: bool,
            all_zones_active: bool,
            basis_purity: bool,
            event_drain: bool,
            all_passed: bool,
        }
        let high_soft = sample_events.iter().any(|e| e.valence > 0.99 && e.grief_load < 1e-4);
        let low_hard = sample_events.iter().any(|e| e.valence < 0.1 && e.grief_load > 0.5);
        let zones_ok = health.zones.len() == n_zones && health.zones.iter().all(|s| s.vectors_processed > 0);
        let purity_ok = health.max_rho < 1e-9;
        let drain_ok = !drained.is_empty();
        let score_ok = health.health_score >= 0.5;
        let all = high_soft && low_hard && zones_ok && purity_ok && drain_ok && score_ok;
        let export = DemoExport {
            health: &health,
            sample_events: &sample_events,
            events_drained: drained.len(),
            gates: DemoGates {
                high_valence_soft: high_soft,
                low_valence_exposure: low_hard,
                all_zones_active: zones_ok,
                basis_purity: purity_ok,
                event_drain: drain_ok,
                all_passed: all,
            },
        };
        match serde_json::to_string_pretty(&export) {
            Ok(s) => println!("{s}"),
            Err(e) => eprintln!("json export failed: {e}"),
        }
        if !all {
            std::process::exit(1);
        }
        return;
    }

    println!("══════════════════════════════════════════════════════════════");
    println!("  Ra-Thor · Soft Feedback Dual-Repo Demo");
    println!("  Ambient ℝ^{AMBIENT_DIM} ⊃ Mercy ℝ^{MERCY_DIM} · Sealed protocol → Powrush-MMO");
    println!("  Contact: info@Rathor.ai");
    println!("══════════════════════════════════════════════════════════════\n");
    println!("  Agents: {n_agents}   Zones: {n_zones}\n");

    println!("──────────────────────────────────────────────────────────────");
    println!("  Sample sealed SoftFeedbackEvent payloads");
    println!("──────────────────────────────────────────────────────────────");
    for ev in &sample_events {
        println!(
            "  {{ zone_id: {}, grief_load: {:.4}, valence: {:.3}, under_floor: {}, tick: {} }}",
            ev.zone_id, ev.grief_load, ev.valence, ev.under_floor, ev.tick
        );
    }

    println!("\n──────────────────────────────────────────────────────────────");
    println!("  LatticeHealthReport");
    println!("──────────────────────────────────────────────────────────────");
    println!("  schema:          {}", health.schema);
    println!("  global_tick:     {}", health.global_tick);
    println!("  total_grief:     {:.3}", health.total_grief);
    println!("  total_vectors:   {}", health.total_vectors);
    println!("  max_rho:         {:.3e}", health.max_rho);
    println!("  total_purify:    {}", health.total_purify_count);
    println!("  max_stress_ema:  {:.4}", health.max_stress_ema);
    println!("  mean_period:     {:.1}", health.mean_effective_period);
    println!("  health_score:    {:.6}", health.health_score);
    println!("  zones H/S/C:     {}/{}/{}", health.zones_healthy, health.zones_stressed, health.zones_critical);
    println!("  healthy:         {}", health.healthy);
    for s in &health.zones {
        println!(
            "    Zone {}: grief={:>10.3}  stress={:>8.4}  vectors={:>6}  purify={:>4}  period={:>5}  ρ={:.3e}  [{}]",
            s.zone_id, s.grief_absorbed, s.stress_ema, s.vectors_processed,
            s.purify_count, s.effective_period, s.last_rho, s.status.as_str()
        );
    }

    println!("\n  Events drained: {}", drained.len());

    let high_soft = sample_events.iter().any(|e| e.valence > 0.99 && e.grief_load < 1e-4);
    let low_hard = sample_events.iter().any(|e| e.valence < 0.1 && e.grief_load > 0.5);
    let zones_ok = health.zones.len() == n_zones && health.zones.iter().all(|s| s.vectors_processed > 0);
    let purity_ok = health.max_rho < 1e-9;

    println!("\n  Verification");
    println!("    High-valence soft path:     {}", if high_soft { "PASS" } else { "FAIL" });
    println!("    Low-valence exposure:       {}", if low_hard { "PASS" } else { "FAIL" });
    println!("    All zones active:           {}", if zones_ok { "PASS" } else { "FAIL" });
    println!("    Basis purity (max ρ):       {}", if purity_ok { "PASS" } else { "FAIL" });
    println!("    Event drain non-empty:      {}", if !drained.is_empty() { "PASS" } else { "FAIL" });
    println!("    Lattice healthy:            {}", if health.healthy { "PASS" } else { "FAIL" });
    let score_ok = health.health_score >= 0.5;
    println!("    Health score ≥ 0.5:         {} ({:.4})", if score_ok { "PASS" } else { "FAIL" }, health.health_score);

    if high_soft && low_hard && zones_ok && purity_ok && !drained.is_empty() && health.healthy && score_ok {
        println!("\n  ★  ALL GATES PASSED — soft feedback dual-repo protocol is live.");
        println!("     Hint: pass --json for machine-readable LatticeHealthReport export.");
    } else {
        println!("\n  ⚠  One or more gates failed.");
        std::process::exit(1);
    }
    println!("\n  Thunder locked. Yoi ⚡\n");
}
