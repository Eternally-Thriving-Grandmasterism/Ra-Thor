//! ONE Organism Full Path Integration Example
//!
//! Demonstrates the hybrid error system including optional miette diagnostics.
//!
//! To enable pretty miette errors:
//!   cargo run --example one_organism_full_path -p self-evolution --features miette
//!
//! Run with: cargo run --example one_organism_full_path -p self-evolution

use self_evolution::{init_sovereign_health_monitor, print_error_chain, BlessingTier};

fn main() {
    println!("=== ONE Organism Full Path + Hybrid Error System ===\n");

    let mut health_monitor = init_sovereign_health_monitor();

    // Trigger an error for demonstration
    let result = health_monitor.load_from_file("nonexistent_state.json");

    match result {
        Ok(_) => println!("Loaded successfully"),
        Err(e) => {
            println!("--- Error Chain (always available) ---");
            print_error_chain(&e);

            // If miette feature is enabled, we can also produce pretty diagnostics
            #[cfg(feature = "miette")]
            {
                println!("\n--- miette Diagnostic Report ---");
                let report = miette::Report::from(e);
                println!("{report:?}");
            }

            #[cfg(not(feature = "miette"))]
            {
                println!("\n(Enable `miette` feature for beautiful diagnostic output)");
            }
        }
    }

    println!("\n=== Done ===");
}