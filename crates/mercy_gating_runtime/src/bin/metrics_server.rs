//! Axum metrics server ready for live SovereignCore state

use axum::{extract::State, routing::get, Json, Router};
use std::sync::Arc;
use tokio::sync::Mutex;

// This would hold a reference to the real SovereignCore in production
pub struct AppState {
    // pub core: Arc<Mutex<SovereignCore>>,
    pub metrics: crate::ArbitrationMetrics,
}

async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "total_events": state.metrics.total_events,
        "proposals_accepted": state.metrics.proposals_accepted,
        "last_event_turn": state.metrics.last_event_turn,
    }))
}

pub async fn start_metrics_server() {
    let state = Arc::new(AppState {
        metrics: crate::ArbitrationMetrics::default(),
    });

    let app = Router::new()
        .route("/metrics", get(get_metrics))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("[METRICS SERVER] Running on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}