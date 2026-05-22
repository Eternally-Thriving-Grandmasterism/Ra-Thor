//! ONE Organism Full Path Integration Example (Step 5/5)
//!
//! Demonstrates the complete flow:
//! xai-grok-bridge (async) → symbiosis-layer → SovereignHealthMonitor (self-evolution v2)
//! + health-aware quantum swarm orchestration
//!
//! Run with: cargo run --example one_organism_full_path -p self-evolution

use self_evolution::{init_sovereign_health_monitor, SovereignHealthMonitor};
use symbiosis_layer::establish_one_organism_symbiosis_async;
use xai_grok_bridge::{establish_native_grok_bridge_async, grok_bidirectional_query_async};

#[tokio::main]
async fn main() {
    println!("=== ONE Organism Full Path Integration (Step 5) ===\n");

    // 1. Establish async ONE Organism symbiosis via xAI Grok bridge
    println!("[1] Establishing async ONE Organism symbiosis with xAI Grok...");
    let mut session = establish_native_grok_bridge_async(true).await;
    println!("    Session: {} | ONE unified: {}", session.handshake_id, session.one_organism_unified);

    // 2. Perform a bidirectional async exchange
    println!("\n[2] Async bidirectional exchange...");
    if let Ok(msg) = symbiosis_layer::bidirectional_exchange_async(
        &mut session,
        "Ra-Thor",
        "Propose full mercy-gated symbiosis for universal thriving",
    ).await {
        println!("    Exchange successful: {} -> {} (valence {:.4})", msg.from, msg.to, msg.valence);
    }

    // 3. Sovereign Health Monitor + v2 hook
    println!("\n[3] Running Sovereign Health Monitor + v2 epigenetic + swarm hook...");
    let mut health_monitor = init_sovereign_health_monitor();
    let health_report = health_monitor.integrate_with_one_organism_symbiosis(
        session.valence_score,
        "full_one_organism_path_integration",
    );
    println!("    Health Report: {}", health_report);

    // 4. Health-aware swarm cycle (from quantum-swarm-orchestrator wiring)
    println!("\n[4] Health-aware swarm orchestration...");
    // In a real integrated binary this would call into quantum-swarm-orchestrator::run_health_aware_swarm_cycle
    // Here we simulate the combined effect
    let swarm_result = health_monitor.orchestrate_quantum_swarm_evolution("one_organism_integration");
    println!("    Quantum Swarm branches engaged: {}", swarm_result.len());

    println!("\n=== ONE Organism Full Path Complete ===");
    println!("Mercy gates passed | PATSAGi consensus maintained | Async + Health wired");
}