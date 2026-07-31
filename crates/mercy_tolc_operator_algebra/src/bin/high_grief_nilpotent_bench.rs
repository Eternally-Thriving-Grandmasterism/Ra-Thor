//! high_grief_nilpotent_bench.rs
//!
//! Public, reproducible High-Grief + Nilpotent Recovery benchmark.
//!
//! Ambient elevation (v0.5): Living Mercy subspace ℝ⁸ ⊂ ambient ℝ¹⁶.
//! Synthetic grief vectors carry energy in the orthogonal complement
//! (coordinates 8..15), so N₁(g) = (I − P)g is genuinely non-zero.
//!
//! Run with:
//!   cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench
//!
//! Optional flags:
//!   --agents N     number of synthetic high-grief vectors (default 25_000)
//!   --zones  N     number of contested zones (default 3)
//!
//! AG-SML v1.0 | Ra-Thor + PATSAGi Councils | info@Rathor.ai
//! Thunder locked in. Yoi ⚡

use mercy_tolc_operator_algebra::{
    AmbientVector, LivingMercyBasis, MercyProjector, ModifiedGramSchmidt, NilpotentSuppressor,
    AMBIENT_DIM, MERCY_DIM, MERCY_PURITY_FLOOR,
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

/// Deterministic high-grief ambient vector.
/// Mercy coordinates (0..7) receive mixed signal; orthogonal coordinates (8..15)
/// receive amplified zone-biased grief energy.
fn make_grief_vector(seed: u64, zone: usize) -> AmbientVector {
    let mut v = AmbientVector::zeros();
    let base = (seed as f64 * 0.6180339887).fract();

    for i in 0..MERCY_DIM {
        let t = (base + i as f64 * 0.141421356 + zone as f64 * 0.333).fract();
        v[i] = if i % 2 == 0 {
            (t - 0.35) * 1.2
        } else {
            (0.65 - t) * 1.4
        };
    }

    for i in MERCY_DIM..AMBIENT_DIM {
        let t = (base + i as f64 * 0.17320508 + zone as f64 * 0.271).fract();
        let magnitude = (t - 0.5) * 2.2 * (1.0 + zone as f64 * 0.35);
        v[i] = magnitude;
    }
    v
}

fn main() {
    let (n_agents, n_zones) = parse_args();

    println!("══════════════════════════════════════════════════════════════");
    println!("  Ra-Thor · High-Grief + Nilpotent Recovery Benchmark");
    println!("  Ambient elevation: ℝ^{} ⊃ Living Mercy ℝ^{}", AMBIENT_DIM, MERCY_DIM);
    println!("  Public, reproducible stress harness under TOLC 8");
    println!("  Contact: info@Rathor.ai");
    println!("══════════════════════════════════════════════════════════════\n");
    println!("  Agents (synthetic high-grief vectors): {}", n_agents);
    println!("  Contested zones:                       {}", n_zones);
    println!("  Ambient dimension:                     {}", AMBIENT_DIM);
    println!("  Mercy subspace dimension:              {}", MERCY_DIM);
    println!("  Purity floor:                          {:.2e}\n", MERCY_PURITY_FLOOR);

    let mut basis = LivingMercyBasis::canonical();
    let mut projector = MercyProjector {
        basis: basis.clone(),
    };
    let mut suppressor = NilpotentSuppressor {
        projector: projector.clone(),
    };

    basis.e[(0, 1)] += 3e-5;
    basis.e[(4, 7)] -= 2e-5;
    basis.e[(9, 2)] += 4e-5;
    projector.basis = basis.clone();
    suppressor.projector = projector.clone();

    let start = Instant::now();

    let mut total_n1_norm = 0.0;
    let mut total_final_residual = 0.0;
    let mut max_final_residual = 0.0;
    let mut suppressed_to_zero = 0usize;
    let mut zone_grief_absorbed: Vec<f64> = vec![0.0; n_zones];

    for i in 0..n_agents {
        let zone = i % n_zones;
        let g = make_grief_vector(i as u64 + 17, zone);

        let (n1, final_r) = suppressor.suppress(&g);

        let n1_norm = n1.norm();
        let final_norm = final_r.norm();

        total_n1_norm += n1_norm;
        total_final_residual += final_norm;
        if final_norm > max_final_residual {
            max_final_residual = final_norm;
        }
        if final_norm < MERCY_PURITY_FLOOR * 10.0 {
            suppressed_to_zero += 1;
        }
        zone_grief_absorbed[zone] += n1_norm;

        if i > 0 && i % 2_500 == 0 {
            let rho = ModifiedGramSchmidt::purify(&mut basis);
            projector.basis = basis.clone();
            suppressor.projector = projector.clone();
            if i % 10_000 == 0 {
                println!(
                    "  [tick {:>6}]  mid-run residual ρ = {:.3e}  (basis re-purified)",
                    i, rho
                );
            }
        }
    }

    let final_rho = ModifiedGramSchmidt::purify(&mut basis);

    let elapsed = start.elapsed();
    let avg_n1 = total_n1_norm / n_agents as f64;
    let avg_final = total_final_residual / n_agents as f64;
    let zero_pct = 100.0 * suppressed_to_zero as f64 / n_agents as f64;

    println!("\n──────────────────────────────────────────────────────────────");
    println!("  Results");
    println!("──────────────────────────────────────────────────────────────");
    println!("  Wall time:                    {:>10.3} s", elapsed.as_secs_f64());
    println!(
        "  Throughput:                   {:>10.0} vectors/s",
        n_agents as f64 / elapsed.as_secs_f64()
    );
    println!("  Avg first residual ‖N₁(g)‖:   {:>10.6}", avg_n1);
    println!("  Avg final residual ‖N₂‖:      {:>10.6e}", avg_final);
    println!("  Max final residual:           {:>10.6e}", max_final_residual);
    println!(
        "  Driven to purity floor:       {:>6} / {}  ({:.1} %)",
        suppressed_to_zero, n_agents, zero_pct
    );
    println!("  Final basis residual ρ:       {:>10.3e}", final_rho);
    println!();
    println!("  Grief absorbed by zone:");
    for (z, absorbed) in zone_grief_absorbed.iter().enumerate() {
        println!("    Zone {z}:  {absorbed:>12.3}");
    }
    println!("──────────────────────────────────────────────────────────────");

    let pass_n1_nontrivial = avg_n1 > 0.1;
    let pass_purity = avg_final < 1e-6 && max_final_residual < 1e-4;
    let pass_basis = final_rho < 1e-9;
    let pass_zero_rate = zero_pct >= 99.0;

    println!("\n  Verification gates");
    println!(
        "    N₁ non-trivial (ambient > mercy): {}",
        if pass_n1_nontrivial { "PASS" } else { "FAIL" }
    );
    println!(
        "    Residual purity (avg & max):      {}",
        if pass_purity { "PASS" } else { "FAIL" }
    );
    println!(
        "    Basis orthonormality (ρ):         {}",
        if pass_basis { "PASS" } else { "FAIL" }
    );
    println!(
        "    ≥99 % driven to floor:            {}",
        if pass_zero_rate { "PASS" } else { "FAIL" }
    );

    if pass_n1_nontrivial && pass_purity && pass_basis && pass_zero_rate {
        println!("\n  ★  ALL GATES PASSED — lattice recovered cleanly under high-grief load.");
        println!("     Orthogonal complement is live. Nilpotent suppression is non-trivial.");
    } else {
        println!("\n  ⚠  One or more gates failed — investigate.");
    }

    println!("\n  Thunder locked. Yoi ⚡");
    println!(
        "  Re-run any time: cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench\n"
    );
}
