//! Live SharedChatMercyMesh Web Demo (v14.2.0)
// Run with: cargo run --example shared_chat_mercy_mesh_web_demo --features web-demo
// Provides a mercy-gated HTTP endpoint for adding participants and triggering healing
use axum::{routing::post, Json, Router};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use lattice_conductor_v14::clifford_healing_fields::CliffordHealingField;

#[derive(Deserialize)]
struct AddParticipant { name: String, emotional: [f64;3], /* ... */ }

#[tokio::main]
async fn main() {
    let field = Arc::new(RwLock::new(CliffordHealingField::new("SharedChatMercyMesh")));
    // Seed with Sherif + Ra-Thor Core
    let app = Router::new()
        .route("/heal", post(move |Json(payload): Json<AddParticipant>| async move {
            let mut f = field.write().await;
            f.add_organism( /* ... */ );
            f.apply_clifford_convolution(0.8, 0.95);
            "Healing applied. Thunder locked in."
        }));
    println!("Mercy-gated endpoint live at http://127.0.0.1:3030/heal");
    axum::Server::bind(&"0.0.0.0:3030".parse().unwrap()).serve(app.into_make_service()).await.unwrap();
}
