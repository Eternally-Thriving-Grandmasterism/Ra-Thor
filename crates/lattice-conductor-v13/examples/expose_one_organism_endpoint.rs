//! Real-time ONE Organism Endpoint with Axum + WebSocket + Dashboard
//! External shards can join and stream real-time state.

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::{Html, IntoResponse},
    routing::{get, get_service},
    Router,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::services::ServeDir;

// Shared broadcast channel for real-time updates to all connected shards
pub type UpdateSender = broadcast::Sender<String>;

async fn websocket_handler(
    ws: WebSocketUpgrade,
    sender: Arc<UpdateSender>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, sender))
}

async fn handle_socket(mut socket: WebSocket, sender: Arc<UpdateSender>) {
    let mut rx = sender.subscribe();

    // Send welcome
    let _ = socket
        .send(Message::Text(json!({
            "type": "welcome",
            "message": "Connected to ONE Organism Endpoint - Rathor.ai + Grok + PATSAGi"
        }).to_string()))
        .await;

    // Listen for incoming messages (shard join requests, etc.)
    let (mut tx, mut rx_ws) = socket.split();

    tokio::spawn(async move {
        while let Some(Ok(msg)) = rx_ws.next().await {
            if let Message::Text(text) = msg {
                // In real impl: parse join request, create SovereignShard, bless it, add to federation
                println!("[Endpoint] Received from shard: {}", text);
                let _ = sender.send(format!("{}" , json!({"type": "shard_message", "data": text})));
            }
        }
    });

    // Forward broadcast updates to this client
    while let Ok(update) = rx.recv().await {
        if tx.send(Message::Text(update)).await.is_err() {
            break;
        }
    }
}

async fn dashboard_handler() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
    <head><title>ONE Organism Dashboard | Ra-Thor v13</title></head>
    <body style="font-family: monospace; background: #0a0a0a; color: #00ff9d;">
        <h1>⚡ ONE Organism Live Dashboard</h1>
        <p><strong>Rathor.ai + Grok + PATSAGi Councils</strong></p>
        <div id="status">Connecting to WebSocket...</div>
        <pre id="log"></pre>
        <script>
            const ws = new WebSocket('ws://localhost:7878/ws');
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                document.getElementById('log').textContent += JSON.stringify(data, null, 2) + '\n';
            };
            ws.onopen = () => { document.getElementById('status').innerHTML = '<span style="color:#00ff9d">CONNECTED</span>'; };
        </script>
    </body>
    </html>
    "#)
}

#[tokio::main]
async fn main() {
    let (tx, _rx) = broadcast::channel::<String>(100);
    let sender = Arc::new(tx);

    let app = Router::new()
        .route("/ws", get(move |ws: WebSocketUpgrade| websocket_handler(ws, sender.clone())))
        .route("/dashboard", get(dashboard_handler))
        .route("/status", get(|| async { "ONE Organism Endpoint Active | Mercy Flowing | Shards Joinable" }))
        .nest_service("/static", get_service(ServeDir::new("static")));

    println!("🚀 ONE Organism Endpoint listening on http://0.0.0.0:7878");
    println!("   Dashboard: http://localhost:7878/dashboard");
    println!("   WebSocket: ws://localhost:7878/ws");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:7878").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}