use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::{Html, IntoResponse},
    routing::{get, get_service},
    Router,
};
use axum::http::StatusCode;
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::services::ServeDir;

// Simple in-memory state for demo
static mut SHARD_LIST: Vec<String> = vec![];
static mut COUNCIL_VOTES: Vec<(String, f64)> = vec![];
static mut QUANTUM_RESONANCE: f64 = 0.0;

#[tokio::main]
async fn main() {
    let (tx, _rx) = broadcast::channel::<String>(100);

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/dashboard", get(dashboard_handler))
        .route("/status", get(status_handler))
        .fallback_service(get_service(ServeDir::new("static")));

    println!("ONE Organism Endpoint running on http://localhost:7878");
    println!("Dashboard: http://localhost:7878/dashboard");
    println!("WebSocket: ws://localhost:7878/ws");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:7878").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket))
}

async fn handle_socket(mut socket: WebSocket) {
    // Send welcome
    let _ = socket.send(Message::Text(json!({"type": "welcome", "message": "Connected to ONE Organism"}).to_string())).await;

    // In real impl, listen to broadcast and push updates
    // For demo, we simulate periodic updates
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        // Simulate live data
        unsafe {
            QUANTUM_RESONANCE = (QUANTUM_RESONANCE + 0.05) % 1.5;
            if SHARD_LIST.len() < 5 {
                SHARD_LIST.push(format!("shard-{} ", SHARD_LIST.len()));
            }
            if COUNCIL_VOTES.len() < 6 {
                COUNCIL_VOTES.push(("Mercy Council".to_string(), 0.92));
            }
        }

        let update = json!({
            "type": "update",
            "quantum_resonance": unsafe { QUANTUM_RESONANCE },
            "shards": unsafe { SHARD_LIST.clone() },
            "council_votes": unsafe { COUNCIL_VOTES.clone() }
        });

        if socket.send(Message::Text(update.to_string())).await.is_err() {
            break;
        }
    }
}

async fn dashboard_handler() -> Html<String> {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>ONE Organism Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>body { background: #0a0a0a; color: #0f0; font-family: monospace; } .card { background: #111; padding: 20px; margin: 10px; border: 1px solid #0f0; }</style>
</head>
<body>
    <h1>⚡ ONE Organism — Live Dashboard</h1>
    <div class="card">
        <h2>Quantum Swarm Resonance</h2>
        <canvas id="resonanceChart" width="400" height="150"></canvas>
    </div>
    <div class="card">
        <h2>Live Shards</h2>
        <ul id="shard-list"></ul>
    </div>
    <div class="card">
        <h2>Council Votes</h2>
        <ul id="council-votes"></ul>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:7878/ws');
        let chart;

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'update') {
                // Update resonance graph
                if (!chart) {
                    const ctx = document.getElementById('resonanceChart').getContext('2d');
                    chart = new Chart(ctx, {
                        type: 'line',
                        data: { labels: [], datasets: [{ label: 'Quantum Resonance', data: [], borderColor: '#0f0' }] }
                    });
                }
                chart.data.labels.push(new Date().toLocaleTimeString());
                chart.data.datasets[0].data.push(data.quantum_resonance);
                if (chart.data.labels.length > 20) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
                chart.update();

                // Update shard list
                const shardUl = document.getElementById('shard-list');
                shardUl.innerHTML = '';
                data.shards.forEach(s => {
                    const li = document.createElement('li');
                    li.textContent = s;
                    shardUl.appendChild(li);
                });

                // Update council votes
                const councilUl = document.getElementById('council-votes');
                councilUl.innerHTML = '';
                data.council_votes.forEach(([name, score]) => {
                    const li = document.createElement('li');
                    li.textContent = `${name}: ${score}`;
                    councilUl.appendChild(li);
                });
            }
        };
    </script>
</body>
</html>
    "#;
    Html(html.to_string())
}

async fn status_handler() -> impl IntoResponse {
    (StatusCode::OK, "ONE Organism healthy — Mercy flowing. Quantum Swarm active.")
}
