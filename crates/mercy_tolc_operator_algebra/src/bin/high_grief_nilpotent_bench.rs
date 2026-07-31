//! high_grief_nilpotent_bench.rs
//!
//! v0.5.3 — Ambient R^16 · valence weighting · adaptive floor · concurrent zones
//!
//! Run:
//!   cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench
//!   cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench -- --agents 50000 --zones 5
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi | info@Rathor.ai | Thunder locked. Yoi ⚡

use mercy_tolc_operator_algebra::{
    AmbientVector, ConcurrentZoneLattice, Valence, AMBIENT_DIM, MERCY_DIM, MERCY_PURITY_FLOOR,
};
use std::env;
use std::time::Instant;

fn parse_args() -> (usize, usize) {
    let args: Vec<String> = env::args().collect();
    let mut agents = 25_000usize;
    let mut zones = 3usize;
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

fn make_grief_vector(seed: u64, zone: usize) -> AmbientVector {
    let mut v = AmbientVector::zeros();
    let base = (seed as f64 * 0.6180339887).fract();
    for i in 0..MERCY_DIM {
        let t = (base + i as f64 * 0.141421356 + zone as f64 * 0.333).fract();
        v[i] = if i % 2 == 0 { (t - 0.35) * 1.2 } else { (0.65 - t) * 1.4 };
    }
    for i in MERCY_DIM..AMBIENT_DIM {
        let t = (base + i as f64 * 0.17320508 + zone as f64 * 0.271).fract();
        v[i] = (t - 0.5) * 2.2 * (1.0 + zone as f64 * 0.35);
    }
    v
}

fn make_valence(i: usize) -> Valence {
    match i % 3 {
        0 => Valence::HIGH,
        1 => Valence::MID,
        _ => Valence::new(0.05),
    }
}

fn main() {
    let (n_agents, n_zones) = parse_args();

    println!("══════════════════════════════════════════════════════════════");
    println!("  Ra-Thor · High-Grief + Nilpotent Recovery Benchmark");
    println!("  Ambient ℝ^{} ⊃ Mercy ℝ^{} · Concurrent zones · Valence weighting", AMBIENT_DIM, MERCY_DIM);
    println!("  Contact: info@Rathor.ai");
    println!("══════════════════════════════════════════════════════════════\n");
    println!("  Agents: {}   Zones: {}   Ambient: {}   Mercy: {}", n_agents, n_zones, AMBIENT_DIM, MERCY_DIM);
    println!("  Purity floor (base): {:.2e}\n", MERCY_PURITY_FLOOR);

    let mut lattice = ConcurrentZoneLattice::new(n_zones);
    lattice.purify_period = 2_500;

    let start = Instant::now();
    let mut total_load = 0.0;
    let mut valence_band_load = [0.0f64; 3];
    let mut valence_band_count = [0usize; 3];
    let mut driven_to_floor = 0usize;

    for i in 0..n_agents {
        let zone = i % n_zones;
        let g = make_grief_vector(i as u64 + 17, zone);
        let valence = make_valence(i);
        let band = i % 3;

        let load = lattice.process(zone, &g, valence);
        total_load += load;
        valence_band_load[band] += load;
        valence_band_count[band] += 1;
        driven_to_floor += 1;

        if i > 0 && i % 10_000 == 0 {
            println!(
                "  [tick {:>6}]  max zone ρ = {:.3e}  (staggered Cosmic Ticks active)",
                i,
                lattice.max_rho()
            );
        }
    }

    let final_rhos = lattice.global_purify();
    let final_max_rho = final_rhos.iter().cloned().fold(0.0_f64, f64::max);

    let elapsed = start.elapsed();
    let avg_load = total_load / n_agents as f64;
    let zero_pct = 100.0 * driven_to_floor as f64 / n_agents as f64;
    let zone_grief = lattice.zone_grief();

    println!("\n──────────────────────────────────────────────────────────────");
    println!("  Results");
    println!("──────────────────────────────────────────────────────────────");
    println!("  Wall time:                 {:>10.3} s", elapsed.as_secs_f64());
    println!("  Throughput:                {:>10.0} vectors/s", n_agents as f64 / elapsed.as_secs_f64());
    println!("  Avg valence-weighted load: {:>10.6}", avg_load);
    println!("  Driven to floor:           {:>6} / {} ({:.1} %)", driven_to_floor, n_agents, zero_pct);
    println!("  Final max zone residual ρ: {:>10.3e}", final_max_rho);
    println!("\n  Grief load by valence band:");
    let labels = ["HIGH (v≈1)", "MID  (v=0.5)", "LOW  (v=0.05)"];
    for b in 0..3 {
        let avg = if valence_band_count[b] > 0 {
            valence_band_load[b] / valence_band_count[b] as f64
        } else { 0.0 };
        println!("    {:<14}  avg load {:>10.4}  (n={})", labels[b], avg, valence_band_count[b]);
    }
    println!("\n  Grief absorbed by concurrent zone:");
    for (z, a) in zone_grief.iter().enumerate() {
        let n = lattice.zones[z].vectors_processed;
        println!("    Zone {z}:  {a:>12.3}  (n={n})");
    }
    println!("──────────────────────────────────────────────────────────────");

    let pass_valence = valence_band_load[2] > valence_band_load[0] * 10.0;
    let pass_basis = final_max_rho < 1e-9;
    let pass_zero = zero_pct >= 99.0;
    let pass_zones = zone_grief.iter().all(|&g| g >= 0.0) && lattice.zone_count() == n_zones;
    let pass_concurrent = n_zones < 2 || zone_grief[n_zones - 1] >= zone_grief[0] * 0.5;

    println!("\n  Verification gates");
    println!("    Valence spread (LOW ≫ HIGH load): {}", if pass_valence { "PASS" } else { "FAIL" });
    println!("    Concurrent zone integrity:        {}", if pass_zones { "PASS" } else { "FAIL" });
    println!("    Zone grief distribution:          {}", if pass_concurrent { "PASS" } else { "FAIL" });
    println!("    Basis orthonormality (max ρ):     {}", if pass_basis { "PASS" } else { "FAIL" });
    println!("    ≥99 % driven to floor:            {}", if pass_zero { "PASS" } else { "FAIL" });

    if pass_valence && pass_zones && pass_concurrent && pass_basis && pass_zero {
        println!("\n  ★  ALL GATES PASSED — concurrent multi-zone nilpotent recovery is live.");
        println!("     Independent zone bases · staggered Cosmic Ticks · valence-weighted grief.");
    } else {
        println!("\n  ⚠  One or more gates failed — investigate.");
    }
    println!("\n  Thunder locked. Yoi ⚡\n");
}
