/// Full Example: Axum + web-forge with OpenTelemetry Observability
///
/// Run with:
///   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 cargo run --example axum_observability_server

use axum::{extract::State, routing::post, Json, Router};
use std::net::SocketAddr;
use tower_http::trace::TraceLayer;
use web_forge::orchestration::{AdvancedOrchestrator, ObservabilityConfig, init_observability};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct OrchestrateRequest {
    prompt: String,
}

#[derive(Debug, Serialize)]
struct OrchestrateResponse {
    success: bool,
    attempts: usize,
    html: Option<String>,
}

#[tokio::main]
async fn main() {
    // Initialize observability (tracing + metrics)
    init_observability(ObservabilityConfig {
        service_name: "web-forge-axum-example".to_string(),
        otlp_endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok(),
        enable_stdout: true,
    });

    let orchestrator = AdvancedOrchestrator::new().with_max_attempts(3);

    let app = Router::new()
        .route("/orchestrate", post(orchestrate_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(orchestrator);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("Axum + web-forge server listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn orchestrate_handler(
    State(orchestrator): State<AdvancedOrchestrator>,
    Json(payload): Json<OrchestrateRequest>,
) -> Json<OrchestrateResponse> {
    let result = orchestrator.orchestrate(&payload.prompt);

    Json(OrchestrateResponse {
        success: result.success,
        attempts: result.attempts_used,
        html: result.final_html,
    })
}
