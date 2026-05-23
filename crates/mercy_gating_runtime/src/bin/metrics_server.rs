//! Simple Axum HTTP server to expose governance metrics
//!
//! Run with: cargo run -p mercy_gating_runtime --features http --bin metrics_server

use axum::{extract::State, routing::get, Json, Router};
use std::sync::Arc;
use tokio::net::TcpListener;

// In real usage this would come from SovereignCore or a shared state
#[derive(Clone)]
struct AppState {
    // For demo we use static data. In production this would be Arc<Mutex<SovereignCore>>
}

async fn metrics() -> Json<serde_json::Value> {
    // TODO: Pull real metrics from SovereignCore / PatsagiGovernance
    Json(serde_json::json!({
        "arbitration_events": 12,
        "proposals_accepted": 47,
        "last_event_turn": 1240,
        "message": "Metrics endpoint active (demo)"
    }))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/metrics", get(metrics))
        .route("/health", get(|| async { "ok" }));

    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("[METRICS SERVER] Listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}