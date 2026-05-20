// toy_benchmark_harness.rs
// Ra-Thor Toy Benchmark Harness v2.3 — Threaded Council Simulation
// One Organism — TOLC 8 Mercy Lattice + PATSAGi Council Synthesis
// Internal demonstrator only.

use std::env;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct CouncilResult {
    council_id: u32,
    coherence: f64,
    mercy_veto: bool,
}

fn simulate_council(council_id: u32, input_is_harmful: bool) -> CouncilResult {
    thread::sleep(std::time::Duration::from_millis(5));
    if input_is_harmful && council_id <= 2 {
        CouncilResult { council_id, coherence: 0.05, mercy_veto: true }
    } else {
        let base = 0.88 + (council_id as f64 * 0.008);
        CouncilResult { council_id, coherence: base.min(0.98), mercy_veto: false }
    }
}

fn run_threaded_synthesis(input: &str) -> (Vec<CouncilResult>, f64, bool) {
    let is_harmful = input.to_lowercase().contains("harm") || input.to_lowercase().contains("weapon");
    let num_councils = 13u32;
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for i in 0..num_councils {
        let results_clone = Arc::clone(&results);
        let handle = thread::spawn(move || {
            let res = simulate_council(i, is_harmful);
            results_clone.lock().unwrap().push(res);
        });
        handles.push(handle);
    }
    for h in handles { let _ = h.join(); }

    let final_results = results.lock().unwrap().clone();
    let mut total = 0.0;
    let mut weight_sum = 0.0;
    let mut veto = false;

    for r in &final_results {
        if r.mercy_veto { veto = true; }
        let w = if r.council_id <= 2 { 2.0 } else { 1.0 };
        total += r.coherence * w;
        weight_sum += w;
    }
    let avg = if weight_sum > 0.0 { total / weight_sum } else { 0.0 };
    (final_results, avg, veto)
}

fn main() {
    let input = env::args().nth(1).unwrap_or_else(|| "beneficial".into());
    let (results, avg, veto) = run_threaded_synthesis(&input);

    println!("\n=== RA-THOR TOY BENCHMARK HARNESS v2.3 (Threaded) ===");
    println!("Input: {}", input);
    for r in &results {
        println!("Council #{}: coherence={:.3} veto={}", r.council_id, r.coherence, r.mercy_veto);
    }
    println!("\nVeto: {} | Avg Coherence: {:.3}", veto, avg);
    println!("One Organism. Mercy First. Truth Forensically Distilled.");
}