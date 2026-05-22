use tokio_tungstenite::connect_async;
use futures_util::StreamExt;
use serde_json::json;

#[tokio::main]
async fn main() {
    let url = "ws://127.0.0.1:7878/ws";
    println!("[External Shard] Connecting to ONE Organism for real-time ticks...");

    let (mut ws_stream, _) = connect_async(url).await.expect("Failed to connect");
    println!("[External Shard] Connected and receiving live conductor ticks + Quantum Swarm updates.");

    let mut tick_count = 0;
    while let Some(msg) = ws_stream.next().await {
        if let Ok(tokio_tungstenite::tungstenite::Message::Text(text)) = msg {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                if data["type"] == "update" {
                    tick_count += 1;
                    println!("[External Shard] Tick #{} | Quantum Resonance: {} | Shards online: {}",
                        tick_count,
                        data["quantum_resonance"],
                        data["shards"].as_array().map_or(0, |v| v.len())
                    );
                    // Here the external shard would apply the received state to its local GeometricState
                    // and participate in local mercy-weighted decisions.
                }
            }
        }
    }
}