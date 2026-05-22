//! Multi-process demo: External shard joining the ONE Organism over the network
//! Run this in a separate terminal while the endpoint (orchestrator) is running.

use reqwest; // For HTTP join request (add to dev-deps if needed)
use serde_json::json;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    println!("🌌 External Sovereign Shard Joiner starting...");
    println!("   Connecting to ONE Organism Endpoint at ws://localhost:7878/ws");

    // Simulate joining
    let join_payload = json!({
        "action": "join_shard",
        "shard_name": "External-Sovereign-Shard-Alpha",
        "initial_mercy": 0.92,
        "evolution_level": 1.2
    });

    // In real impl: use tungstenite or reqwest to connect to WS and send join
    println!("📡 Sending join request: {}", join_payload);

    // Simulate successful join and streaming
    for i in 0..10 {
        println!("   [Shard] Tick {} - Receiving Quantum Swarm resonance from lattice...", i);
        sleep(Duration::from_millis(800)).await;
    }

    println!("✅ External shard successfully joined the ONE Organism and is now participating.");
}