use tokio_tungstenite::connect_async;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;

#[tokio::main]
async fn main() {
    let url = "ws://127.0.0.1:7878/ws";
    println!("External shard attempting to join ONE Organism at {}", url);

    let (mut ws_stream, _) = connect_async(url).await.expect("Failed to connect");
    println!("Connected to ONE Organism endpoint!");

    // Send join request
    let join_msg = json!({
        "type": "join_request",
        "shard_id": "external-shard-001",
        "mercy_alignment": 0.94
    });
    ws_stream.send(tokio_tungstenite::tungstenite::Message::Text(join_msg.to_string())).await.unwrap();

    // Listen for updates
    while let Some(msg) = ws_stream.next().await {
        match msg {
            Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                println!("Received from ONE Organism: {}", text);
            }
            _ => {}
        }
    }
}