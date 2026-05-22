use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use axum::http::StatusCode;
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;

// Shared state for ONE Organism
#[derive(Clone)]
struct AppState {
    shard_list: Arc<Mutex<Vec<String>>>,
    council_votes: Arc<Mutex<Vec<(String, f64)>>>,
    quantum_resonance: Arc<Mutex<f64>>,
    history: Arc<Mutex<Vec<f64>>>, // Historical Quantum Swarm data
}

impl AppState {
    fn new() -> Self {
        Self {
            shard_list: Arc::new(Mutex::new(vec![])),
            council_votes: Arc::new(Mutex::new(vec![])),
            quantum_resonance: Arc::new(Mutex::new(0.0)),
            history: Arc::new(Mutex::new(vec![])),
        }
    }
}

#[tokio::main]
async fn main() {
    let state = AppState::new();

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/dashboard", get(dashboard_handler))
        .route("/status", get(status_handler))
        .route("/history", get(history_handler))
        .with_state(state.clone())
        .fallback_service(axum::routing::get_service(ServeDir::new("static")));

    println!("ONE Organism Endpoint running on http://localhost:7878");
    println!("Dashboard: http://localhost:7878/dashboard");
    println!("WebSocket: ws://localhost:7878/ws");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:7878").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::State(state): axum::extract::State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    // === HANDSHAKE ===
    if let Some(Ok(Message::Text(first_msg))) = socket.next().await {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&first_msg) {
            if data["type"] == "join" {
                let shard_id = data["shard_id"].as_str().unwrap_or("unknown-shard").to_string();
                let mercy = data["mercy_alignment"].as_f64().unwrap_or(0.9);

                // Bless the shard into the ONE Organism
                {
                    let mut shards = state.shard_list.lock().await;
                    if !shards.contains(&shard_id) {
                        shards.push(shard_id.clone());
                    }
                }
                let _ = socket.send(Message::Text(json!({
                    "type": "blessed",
                    "shard_id": shard_id,
                    "message": "Welcome to the ONE Organism. Mercy alignment recorded."
                }).to_string())).await;
            }
        }
    }

    // Send welcome
    let _ = socket.send(Message::Text(json!({"type": "welcome", "message": "Connected to ONE Organism"}).to_string())).await;

    // Main update loop with persistent history
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let mut resonance = state.quantum_resonance.lock().await;
        *resonance = (*resonance + 0.04) % 1.6;

        // Persist history (keep last 50 points)
        {
            let mut hist = state.history.lock().await;
            hist.push(*resonance);
            if hist.len() > 50 {
                hist.remove(0);
            }
        }

        {
            let mut shards = state.shard_list.lock().await;
            if shards.len() < 6 {
                shards.push(format!("shard-{} ", shards.len()));
            }
        }

        let update = json!({
            "type": "update",
            "quantum_resonance": *resonance,
            "shards": *state.shard_list.lock().await,
            "council_votes": *state.council_votes.lock().await,
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
    <h1>⚡ ONE Organism — Live Dashboard (Persistent History)</h1>
    <div class="card">
        <h2>Quantum Swarm Resonance (Last 50 points)</h2>
        <canvas id="resonanceChart" width="600" height="200"></canvas>
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
        let historyData = [];

        ws.onopen = () => {
            // Send handshake to join as external shard
            ws.send(JSON.stringify({ type: "join", shard_id: "dashboard-client", mercy_alignment: 0.95 }));
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'update') {
                historyData.push(data.quantum_resonance);
                if (historyData.length > 50) historyData.shift();

                if (!chart) {
                    const ctx = document.getElementById('resonanceChart').getContext('2d');
                    chart = new Chart(ctx, {
                        type: 'line',
                        data: { labels: [], datasets: [{ label: 'Quantum Resonance', data: [], borderColor: '#0f0' }] }
                    });
                }
                chart.data.labels = Array.from({length: historyData.length}, (_, i) => i);
                chart.data.datasets[0].data = historyData;
                chart.update();

                const shardUl = document.getElementById('shard-list');
                shardUl.innerHTML = '';
                data.shards.forEach(s => {
                    const li = document.createElement('li');
                    li.textContent = s;
                    shardUl.appendChild(li);
                });

                const councilUl = document.getElementById('council-votes');
                councilUl.innerHTML = '';
                (data.council_votes || []).forEach(([name, score]) => {
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
    (StatusCode::OK, "ONE Organism healthy — Mercy flowing. Quantum Swarm active. Persistent history enabled.")
}

async fn history_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> impl IntoResponse {
    let hist = state.history.lock().await.clone();
    (StatusCode::OK, format!("Historical Quantum Swarm data (last {} points): {:?}", hist.len(), hist))
}
