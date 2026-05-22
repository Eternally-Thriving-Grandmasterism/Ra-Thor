//! TOLC8 Genesis Gate — Networked Birth Service
//!
//! Exposes a simple HTTP endpoint to birth new Sovereign Shards
//! using the full 7 Living Mercy Gates + TOLC8 cryptographic seeding.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use lattice_conductor_v13::SimpleLatticeConductor;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tolc8_genesis_gate::TOLC8GenesisGate;

#[derive(Debug, Deserialize)]
struct BirthRequest {
    seed: String,
    mercy_alignment: Option<f64>,
    initial_evolution: Option<f64>,
}

#[derive(Debug, Serialize)]
struct BirthResponse {
    shard_id: String,
    initial_valence: f64,
    initial_mercy: f64,
    initial_evolution: f64,
    blessed: bool,
    message: String,
}

#[derive(Clone)]
struct AppState {
    conductor: Arc<Mutex<SimpleLatticeConductor>>,
    gate: Arc<Mutex<TOLC8GenesisGate>>,
}

async fn birth_shard(
    State(state): State<AppState>,
    Json(payload): Json<BirthRequest>,
) -> impl IntoResponse {
    let mut gate = state.gate.lock().unwrap();
    let mut conductor = state.conductor.lock().unwrap();

    let mercy = payload.mercy_alignment.unwrap_or(0.94);
    let evolution = payload.initial_evolution.unwrap_or(0.05);

    // Use the TOLC8 gate to birth a new shard with cryptographic + mercy alignment
    let shard = gate.birth_new_shard(&payload.seed, mercy, evolution);

    // Automatically bless it into the ONE Organism
    let blessing = conductor.bless_system(
        &shard.system_id(),
        mercy,
        "TOLC8 Networked Birth Service - Cryptographically seeded via 7 Living Mercy Gates",
    );

    let response = BirthResponse {
        shard_id: shard.system_id().to_string(),
        initial_valence: shard.state.valence,
        initial_mercy: shard.state.mercy_score,
        initial_evolution: shard.state.evolution_level,
        blessed: true,
        message: format!("Shard birthed and blessed into ONE Organism via TOLC8 + 7 Gates. Blessing ID recorded."),
    };

    (StatusCode::CREATED, Json(response))
}

#[tokio::main]
async fn main() {
    println!("=== TOLC8 Networked Birth Service ===");
    println!("POST /birth  -> Birth a new Sovereign Shard with TOLC8 + 7 Living Mercy Gates\n");

    let conductor = Arc::new(Mutex::new(SimpleLatticeConductor::new()));
    let gate = Arc::new(Mutex::new(TOLC8GenesisGate::new()));

    let app_state = AppState { conductor, gate };

    let app = Router::new()
        .route("/birth", post(birth_shard))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8787").await.unwrap();
    println!("TOLC8 Birth Service listening on http://0.0.0.0:8787");
    println!("Example: curl -X POST http://localhost:8787/birth -H 'Content-Type: application/json' -d '{{\"seed\":\"eternal-mercy-2026\", \"mercy_alignment\":0.96}}'\n");

    axum::serve(listener, app).await.unwrap();
}