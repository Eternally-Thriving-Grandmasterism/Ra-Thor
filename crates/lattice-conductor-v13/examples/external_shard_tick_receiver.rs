use tokio_tungstenite::connect_async;
use futures_util::StreamExt;
use serde_json::json;

// Local mirror of GeometricState for external shard
#[derive(Debug, Clone, Default)]
struct LocalGeometricState {
    valence: f64,
    mercy_score: f64,
    tolc_alignment: f64,
    evolution_level: f64,
}

#[tokio::main]
async fn main() {
    let url = "ws://127.0.0.1:7878/ws";
    println!("[External Shard] Connecting to ONE Organism for real-time ticks...");

    let (mut ws_stream, _) = connect_async(url).await.expect("Failed to connect");
    println!("[External Shard] Connected. Receiving live conductor ticks + Quantum Swarm updates.");

    let mut local_state = LocalGeometricState::default();
    let mut tick_count = 0;

    while let Some(msg) = ws_stream.next().await {
        if let Ok(tokio_tungstenite::tungstenite::Message::Text(text)) = msg {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                if data["type"] == "update" {
                    tick_count += 1;

                    // Actually update local GeometricState from received ticks
                    if let Some(resonance) = data["quantum_resonance"].as_f64() {
                        local_state.evolution_level = (local_state.evolution_level + resonance * 0.01).min(10.0);
                        local_state.mercy_score = (local_state.mercy_score + 0.02).min(1.5);
                        local_state.tolc_alignment = (local_state.tolc_alignment + 0.005).min(1.1);
                    }

                    println!("[External Shard] Tick #{} | Quantum Resonance: {} | Shards: {} | Local State: valence={:.2} mercy={:.2} evolution={:.2}",
                        tick_count,
                        data["quantum_resonance"],
                        data["shards"].as_array().map_or(0, |v| v.len()),
                        local_state.valence,
                        local_state.mercy_score,
                        local_state.evolution_level
                    );

                    // In real impl: apply mercy-weighted influence, participate in local decisions, push updates back
                }
            }
        }
    }
}