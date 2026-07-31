//! high_grief_nilpotent_bench.rs
//!
//! Ambient elevation (v0.5) + valence (v0.5.1) + adaptive purity floor (v0.5.2):
//! Living Mercy ℝ⁸ ⊂ ambient ℝ¹⁶. Orthogonal residual scaled by (1 − valence).
//!
//! Run:
//!   cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench
//!
//! Flags: --agents N  --zones N
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi | info@Rathor.ai | Thunder locked. Yoi ⚡

use mercy_tolc_operator_algebra::{
    AmbientVector, LivingMercyBasis, MercyProjector, ModifiedGramSchmidt, NilpotentSuppressor,
    Valence, AMBIENT_DIM, MERCY_DIM, MERCY_PURITY_FLOOR,
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
    println!("  Ambient elevation: ℝ^{} ⊃ Living Mercy ℝ^{}", AMBIENT_DIM, MERCY_DIM);
    println!("  Valence weighting: grief_load = (1−v)·‖(I−P)g‖");
    println!("  Contact: info@Rathor.ai");
    println!("══════════════════════════════════════════════════════════════\n");
    println!("  Agents: {}   Zones: {}   Ambient: {}   Mercy: {}", n_agents, n_zones, AMBIENT_DIM, MERCY_DIM);
    println!("  Purity floor: {:.2e}\n", MERCY_PURITY_FLOOR);

    let mut basis = LivingMercyBasis::canonical();
    let mut projector = MercyProjector { basis: basis.clone() };
    let mut suppressor = NilpotentSuppressor { projector: projector.clone() };

    basis.e[(0, 1)] += 3e-5;
    basis.e[(4, 7)] -= 2e-5;
    basis.e[(9, 2)] += 4e-5;
    projector.basis = basis.clone();
    suppressor.projector = projector.clone();

    let start = Instant::now();
    let mut total_raw_n1 = 0.0;
    let mut total_grief_load = 0.0;
    let mut total_final = 0.0;
    let mut max_final = 0.0;
    let mut suppressed_to_zero = 0usize;
    let mut zone_grief: Vec<f64> = vec![0.0; n_zones];
    let mut valence_band_load = [0.0f64; 3];
    let mut valence_band_count = [0usize; 3];

    for i in 0..n_agents {
        let zone = i % n_zones;
        let g = make_grief_vector(i as u64 + 17, zone);
        let valence = make_valence(i);
        let band = i % 3;

        let (raw_n1, _w, final_r, grief_load, _under) =
            suppressor.suppress_weighted(&g, valence);

        total_raw_n1 += raw_n1.norm();
        total_grief_load += grief_load;
        let final_norm = final_r.norm();
        total_final += final_norm;
        if final_norm > max_final { max_final = final_norm; }
        if final_norm < MERCY_PURITY_FLOOR * 10.0 { suppressed_to_zero += 1; }
        zone_grief[zone] += grief_load;
        valence_band_load[band] += grief_load;
        valence_band_count[band] += 1;

        if i > 0 && i % 2_500 == 0 {
            let rho = ModifiedGramSchmidt::purify(&mut basis);
            projector.basis = basis.clone();
            suppressor.projector = projector.clone();
            if i % 10_000 == 0 {
                println!("  [tick {:>6}]  mid-run residual ρ = {:.3e}", i, rho);
            }
        }
    }

    let final_rho = ModifiedGramSchmidt::purify(&mut basis);
    let elapsed = start.elapsed();
    let avg_raw = total_raw_n1 / n_agents as f64;
    let avg_load = total_grief_load / n_agents as f64;
    let avg_final = total_final / n_agents as f64;
    let zero_pct = 100.0 * suppressed_to_zero as f64 / n_agents as f64;

    println!("\n──────────────────────────────────────────────────────────────");
    println!("  Results");
    println!("──────────────────────────────────────────────────────────────");
    println!("  Wall time:                 {:>10.3} s", elapsed.as_secs_f64());
    println!("  Throughput:                {:>10.0} vectors/s", n_agents as f64 / elapsed.as_secs_f64());
    println!("  Avg raw residual ‖N₁‖:     {:>10.6}", avg_raw);
    println!("  Avg valence-weighted load: {:>10.6}", avg_load);
    println!("  Avg final residual ‖N₂‖:   {:>10.6e}", avg_final);
    println!("  Max final residual:        {:>10.6e}", max_final);
    println!("  Driven to floor:           {:>6} / {} ({:.1} %)", suppressed_to_zero, n_agents, zero_pct);
    println!("  Final basis residual ρ:    {:>10.3e}", final_rho);
    println!("\n  Grief load by valence band:");
    let labels = ["HIGH (v≈1)", "MID  (v=0.5)", "LOW  (v=0.05)"];
    for b in 0..3 {
        let avg = if valence_band_count[b] > 0 {
            valence_band_load[b] / valence_band_count[b] as f64
        } else { 0.0 };
        println!("    {:<14}  avg load {:>10.4}  (n={})", labels[b], avg, valence_band_count[b]);
    }
    println!("\n  Grief absorbed by zone (weighted):");
    for (z, a) in zone_grief.iter().enumerate() {
        println!("    Zone {z}:  {a:>12.3}");
    }
    println!("──────────────────────────────────────────────────────────────");

    let pass_n1 = avg_raw > 0.1;
    let pass_valence = valence_band_load[2] > valence_band_load[0] * 10.0;
    let pass_purity = avg_final < 1e-6 && max_final < 1e-4;
    let pass_basis = final_rho < 1e-9;
    let pass_zero = zero_pct >= 99.0;

    println!("\n  Verification gates");
    println!("    N₁ non-trivial:                   {}", if pass_n1 { "PASS" } else { "FAIL" });
    println!("    Valence spread (LOW ≫ HIGH load): {}", if pass_valence { "PASS" } else { "FAIL" });
    println!("    Residual purity:                  {}", if pass_purity { "PASS" } else { "FAIL" });
    println!("    Basis orthonormality (ρ):         {}", if pass_basis { "PASS" } else { "FAIL" });
    println!("    ≥99 % driven to floor:            {}", if pass_zero { "PASS" } else { "FAIL" });

    if pass_n1 && pass_valence && pass_purity && pass_basis && pass_zero {
        println!("\n  ★  ALL GATES PASSED — valence-weighted nilpotent recovery is live.");
        println!("     High valence softens grief. Low valence exposes full orthogonal load.");
    } else {
        println!("\n  ⚠  One or more gates failed — investigate.");
    }
    println!("\n  Thunder locked. Yoi ⚡\n");
}
