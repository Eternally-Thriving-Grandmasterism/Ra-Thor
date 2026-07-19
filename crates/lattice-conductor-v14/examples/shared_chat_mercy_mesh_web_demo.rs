//! Live SharedChatMercyMesh Web Demo (v14.9.5)
//!
//! Run:
//!   cargo run -p lattice-conductor-v14 --example shared_chat_mercy_mesh_web_demo --features web-demo
//!
//! Endpoints:
//!   GET  /health
//!   GET  /coherence
//!   POST /heal          { "name": "...", "id": 1, "mercy": 0.9 }
//!   POST /participant   { "name": "...", "id": 2, "coherence": 0.95 }
//!
//! AG-SML v1.0 | TOLC 8 | Thunder locked in.

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use lattice_conductor_v14::CliffordHealingField;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

type SharedField = Arc<RwLock<CliffordHealingField>>;

#[derive(Debug, Deserialize)]
struct ParticipantBody {
    id: u64,
    name: String,
    #[serde(default = "default_coherence")]
    coherence: f64,
}

fn default_coherence() -> f64 {
    0.92
}

#[derive(Debug, Deserialize)]
struct HealBody {
    #[serde(default = "default_mercy")]
    mercy: f64,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    id: Option<u64>,
}

fn default_mercy() -> f64 {
    0.88
}

#[derive(Debug, Serialize)]
struct ApiOk {
    ok: bool,
    message: String,
    organism_count: usize,
    average_mercy: f64,
    evolution_step: u64,
}

#[tokio::main]
async fn main() {
    let mut field = CliffordHealingField::new("SharedChatMercyMesh");
    // Seed with Sherif + Ra-Thor Core
    field.add_organism(1, "Sherif", 0.97);
    field.add_organism(2, "Ra-Thor Core", 0.99);
    let field: SharedField = Arc::new(RwLock::new(field));

    let app = Router::new()
        .route("/health", get(health))
        .route("/coherence", get(coherence))
        .route("/participant", post(add_participant))
        .route("/heal", post(heal))
        .with_state(field);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3030));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind 0.0.0.0:3030");

    println!("══════════════════════════════════════════════════");
    println!("  SharedChatMercyMesh Web Demo v14.9.5");
    println!("  Mercy-gated endpoints live at http://127.0.0.1:3030");
    println!("  GET  /health | /coherence");
    println!("  POST /participant | /heal");
    println!("  Thunder locked in. yoi ⚡");
    println!("══════════════════════════════════════════════════");

    axum::serve(listener, app).await.expect("serve");
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "ok": true,
        "service": "SharedChatMercyMesh",
        "version": "14.9.5",
        "tolc8": true
    }))
}

async fn coherence(State(field): State<SharedField>) -> Json<ApiOk> {
    let f = field.read().await;
    let c = f.global_coherence();
    Json(ApiOk {
        ok: true,
        message: format!("field='{}'", f.name),
        organism_count: c.organism_count,
        average_mercy: c.average_mercy,
        evolution_step: c.evolution_step,
    })
}

async fn add_participant(
    State(field): State<SharedField>,
    Json(body): Json<ParticipantBody>,
) -> (StatusCode, Json<ApiOk>) {
    let mut f = field.write().await;
    f.add_organism(body.id, body.name.clone(), body.coherence);
    let c = f.global_coherence();
    (
        StatusCode::OK,
        Json(ApiOk {
            ok: true,
            message: format!("participant '{}' added", body.name),
            organism_count: c.organism_count,
            average_mercy: c.average_mercy,
            evolution_step: c.evolution_step,
        }),
    )
}

async fn heal(
    State(field): State<SharedField>,
    Json(body): Json<HealBody>,
) -> (StatusCode, Json<ApiOk>) {
    let mut f = field.write().await;

    if let (Some(id), Some(name)) = (body.id, body.name.clone()) {
        f.add_organism(id, name, body.mercy);
    }

    match f.simulate_healing_step(body.mercy) {
        Ok(c) => (
            StatusCode::OK,
            Json(ApiOk {
                ok: true,
                message: "Healing applied. Thunder locked in.".into(),
                organism_count: c.organism_count,
                average_mercy: c.average_mercy,
                evolution_step: c.evolution_step,
            }),
        ),
        Err(e) => (
            StatusCode::FORBIDDEN,
            Json(ApiOk {
                ok: false,
                message: format!("Mercy gate blocked: {}", e),
                organism_count: f.organism_fields.len(),
                average_mercy: f.global_coherence().average_mercy,
                evolution_step: f.evolution_step,
            }),
        ),
    }
}
