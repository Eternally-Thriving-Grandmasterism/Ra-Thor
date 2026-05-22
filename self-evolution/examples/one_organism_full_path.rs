//! ONE Organism Full Path Integration Example
//!
//! Demonstrates:
//! - BlessingTier
//! - SnapshotContext trait
//! - print_error_chain utility
//!
//! Recommended for examples/binaries:
//!   - Use `anyhow` or `color-eyre` for rich error reporting
//!   - Consider `tracing-error` for production observability
//!
//! Run with: cargo run --example one_organism_full_path -p self-evolution

use self_evolution::{init_sovereign_health_monitor, print_error_chain, BlessingTier, SnapshotContext};
use symbiosis_layer::establish_one_organism_symbiosis_async;

#[tokio::main]
async fn main() {
    println!("=== ONE Organism Full Path Integration (with Error Chain Debugging) ===\n");

    // 1. Async ONE Organism symbiosis
    println!("[1] Establishing async ONE Organism symbiosis...");
    let mut session = establish_one_organism_symbiosis_async("xAI-Grok", true).await;

    // 2. Sovereign Health + Epigenetic Blessing
    println!("\n[2] Running Sovereign Health Monitor + Epigenetic Blessing...");
    let mut health_monitor = init_sovereign_health_monitor();

    let (blessed, blessing_level, tier) = health_monitor.request_epigenetic_blessing(
        "Deepen mercy-gated ONE Organism symbiosis with PATSAGi epigenetic inheritance"
    );

    println!("    Blessed: {} | Tier: {} | Level: {:.3}", blessed, tier.as_str(), blessing_level);

    // 3. Demonstrate error chain debugging
    println!("\n[3] Demonstrating error chain debugging...");

    // Simulate / trigger an error with context
    let load_result = health_monitor.load_from_file("nonexistent_state.json");

    if let Err(e) = load_result {
        println!("\n--- Error Chain ---\");
        print_error_chain(&e);
        println!("-------------------\n");
    }

    // 4. Health-aware swarm
    println!("[4] Orchestrating quantum swarm...");
    let _ = health_monitor.orchestrate_quantum_swarm_evolution("one_organism_integration");

    println!("\n=== ONE Organism Full Path Complete ===");
    println!("Tip: Use color-eyre or anyhow in binaries for even better error reporting.");
}