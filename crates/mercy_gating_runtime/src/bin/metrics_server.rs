//! Axum server with live metrics + reputation exposure

use axum::{extract::State, routing::get, Json, Router};
use std::sync::Arc;

pub struct AppState {
    pub metrics: crate::ArbitrationMetrics,
    // TODO: Add reputations: HashMap<u32, CouncilReputation> when fully wired
}

async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "arbitration_metrics": {
            "total_events": state.metrics.total_events,
            "proposals_accepted": state.metrics.proposals_accepted,
            "last_event_turn": state.metrics.last_event_turn,
        },
        "reputation": {
            "example_council_recent_success_rate": 0.82,
            "note": "Connect SovereignCore for live per-council reputation data including recent_success_rate"
        }
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
    println!("[METRICS SERVER] Listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}