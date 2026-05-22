//! ONE Organism Full Path Integration Example
//!
//! This example demonstrates the hybrid error handling system in `self-evolution`:
//!
//! - Error chaining with `print_error_chain`
//! - Context attachment via `SnapshotContext`
//! - Optional pretty diagnostics with `miette`
//!
//! ## Run
//!
//! ```bash
//! # Basic run
//! cargo run --example one_organism_full_path -p self-evolution
//!
//! # With pretty miette diagnostics
//! cargo run --example one_organism_full_path -p self-evolution --features miette
//! ```
//!
//! AG-SML v1.0

use self_evolution::{init_sovereign_health_monitor, print_error_chain};

fn main() {
    println!("=== ONE Organism + Hybrid Error System Demo ===\n");

    let mut monitor = init_sovereign_health_monitor();

    // === Scenario 1: File not found ===
    println!("[1] Trying to load a non-existent file...\n");
    if let Err(e) = monitor.load_from_file("nonexistent_state.json") {
        print_error_chain(&e);

        #[cfg(feature = "miette")]
        {
            println!("\n--- miette Diagnostic Report ---");
            let report = miette::Report::from(e);
            eprintln!("{report:?}");
        }
    }

    println!("\n=== Demo Complete ===");
}