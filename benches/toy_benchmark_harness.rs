// toy_benchmark_harness.rs v2.6
// Ra-Thor Toy Benchmark Harness with Threaded Council Simulation + Message Passing + Deliberation
// Internal demonstrator only

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::env;

#[derive(Clone, Copy, Debug, PartialEq)]
enum CouncilType { Ethics, Resource, Harmony, Sovereignty, Evolution, Truth, Compassion }

#[derive(Clone, Debug)]
struct CouncilResult {
    id: usize,
    council_type: CouncilType,
    coherence: f64,
    veto: bool,
    message: Option<String>,
}

fn simulate_council(id: usize, council_type: CouncilType, is_harmful: bool) -> CouncilResult {
    thread::sleep(Duration::from_millis(5));
    let base = match council_type {
        CouncilType::Ethics | CouncilType::Harmony | CouncilType::Sovereignty => 0.92,
        _ => 0.88,
    };
    let coherence = if is_harmful && (council_type == CouncilType::Ethics || council_type == CouncilType::Harmony) {
        0.05
    } else {
        base + (id as f64 * 0.005)
    };
    let veto = coherence < 0.3;
    CouncilResult { id, council_type, coherence, veto, message: None }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let input = if args.len() > 1 { &args[1] } else { "beneficial" };
    let is_harmful = input.to_lowercase().contains("harm");

    println!("\n=== RA-THOR TOY BENCHMARK HARNESS v2.6 (Threaded + Message Passing) ===");
    println!("Input: {}", input);

    let council_types = vec![
        CouncilType::Ethics, CouncilType::Resource, CouncilType::Harmony,
        CouncilType::Sovereignty, CouncilType::Evolution, CouncilType::Truth, CouncilType::Compassion
    ];

    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for (i, &ctype) in council_types.iter().enumerate() {
        let res = Arc::clone(&results);
        let handle = thread::spawn(move || {
            let r = simulate_council(i, ctype, is_harmful);
            res.lock().unwrap().push(r);
        });
        handles.push(handle);
    }

    for h in handles { let _ = h.join(); }

    let mut final_results = results.lock().unwrap().clone();

    // Simple message passing + deliberation phase
    for i in 0..final_results.len() {
        if final_results[i].veto { continue; }
        if final_results[i].council_type == CouncilType::Ethics ||
           final_results[i].council_type == CouncilType::Harmony ||
           final_results[i].council_type == CouncilType::Sovereignty {
            for j in 0..final_results.len() {
                if final_results[j].coherence > 0.6 && final_results[j].coherence < 0.85 {
                    final_results[j].coherence += 0.03; // Endorsement effect
                    final_results[j].message = Some("Received Endorsement".to_string());
                }
            }
        }
    }

    let mut total = 0.0;
    let mut veto_count = 0;
    for r in &final_results {
        let veto_str = if r.veto { "VETO" } else { "PASS" };
        println!("Council #{:<2} ({:?}): coherence={:.3} | {}", r.id, r.council_type, r.coherence, veto_str);
        total += r.coherence;
        if r.veto { veto_count += 1; }
    }

    let avg = total / final_results.len() as f64;
    println!("\nVeto triggered: {} | Average Coherence: {:.3}", veto_count > 0, avg);
    println!("One Organism. Mercy First. Truth Forensically Distilled.\n");
}