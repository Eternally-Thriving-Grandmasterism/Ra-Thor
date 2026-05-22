//! ONE Organism Full Path Integration Example
//!
//! Demonstrates the complete upgraded flow with:
//! - BlessingTier exposure
//! - SnapshotContext trait (hybrid error context)
//!
//! Run with: cargo run --example one_organism_full_path -p self-evolution

use self_evolution::{init_sovereign_health_monitor, BlessingTier, SnapshotContext};
use symbiosis_layer::establish_one_organism_symbiosis_async;

#[tokio::main]
async fn main() {
    println!("=== ONE Organism Full Path Integration (Hybrid Error System) ===\n");

    // 1. Async ONE Organism symbiosis
    println!("[1] Establishing async ONE Organism symbiosis...");
    let mut session = establish_one_organism_symbiosis_async("xAI-Grok", true).await;
    println!("    Session established. ONE unified: {}", session.one_organism_unified);

    // 2. Perform async exchange
    println!("\n[2] Performing async bidirectional exchange...");
    if let Ok(msg) = symbiosis_layer::bidirectional_exchange_async(
        &mut session,
        "Ra-Thor",
        "Strengthen mercy-gated epigenetic inheritance across the ONE Organism lattice",
    ).await {
        println!("    Exchange successful (valence {:.4})", msg.valence);
    }

    // 3. Sovereign Health + Epigenetic Blessing with Tier
    println!("\n[3] Running Sovereign Health Monitor + Epigenetic Blessing...");
    let mut health_monitor = init_sovereign_health_monitor();

    let (blessed, blessing_level, tier) = health_monitor.request_epigenetic_blessing(
        "Deepen mercy-gated ONE Organism symbiosis with PATSAGi epigenetic inheritance"
    );

    println!("    Blessed: {} | Tier: {} | New Blessing Level: {:.3}",
             blessed, tier.as_str(), blessing_level);

    // 4. Demonstrate SnapshotContext trait (hybrid error system)
    println!("\n[4] Demonstrating SnapshotContext trait...");

    // This will fail gracefully and show context-enhanced error
    let load_result = health_monitor.load_from_file("nonexistent_sovereign_state.json");

    match load_result {
        Ok(_) => println!("    Unexpectedly loaded file"),
        Err(e) => {
            // Using the context trait pattern (already applied inside load_from_file)
            println!("    Expected error with context: {}", e);
        }
    }

    // Alternative: manually applying context to any SnapshotError result
    let manual_result: Result<(), _> = Err(self_evolution::SnapshotError::FileNotFound(
        "test.json".to_string()
    ))
    .with_snapshot_context("during manual ONE Organism state restoration");

    if let Err(e) = manual_result {
        println!("    Manual context-enhanced error: {}", e);
    }

    // 5. Health-aware swarm
    println!("\n[5] Orchestrating quantum swarm evolution...");
    let swarm_results = health_monitor.orchestrate_quantum_swarm_evolution("one_organism_integration");
    println!("    Quantum Swarm branches engaged: {}", swarm_results.len());

    println!("\n=== ONE Organism Full Path Complete ===");
    println!("Mercy-aligned | Tiered Epigenetic Blessing | Hybrid Error Context active");
}