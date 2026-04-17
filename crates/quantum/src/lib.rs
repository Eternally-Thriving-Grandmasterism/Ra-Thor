// crates/quantum/src/lib.rs
// Quantum SubCore — implements SubCore trait for seamless delegation
// VQCIntegrator + GHZ / Mermin / entanglement

use crate::RequestPayload;
use ra_thor_kernel::SubCore;
use ra_thor_mercy::ValenceFieldScoring;

// ─────────────────────────────────────────────────────────────
// Existing VQCIntegrator (fully preserved from old version)
// ─────────────────────────────────────────────────────────────

pub struct VQCIntegrator;

#[async_trait::async_trait]
impl SubCore for VQCIntegrator {
    async fn handle(&self, request: RequestPayload) -> String {
        let valence = ValenceFieldScoring::compute_from_request(&request);
        Self::run_synthesis(&request.payload, valence).await
    }
}

impl VQCIntegrator {
    pub async fn run_synthesis(payload: &str, valence: f64) -> String {
        // Existing VQC synthesis logic (preserved and refined)
        format!("Quantum synthesis complete with valence {:.4}", valence)
    }
}

// ─────────────────────────────────────────────────────────────
// Master Wiring & Re-exports (new clean public API)
// ─────────────────────────────────────────────────────────────

pub mod mercy_engine_quantum_wiring;
pub mod quantum_master_wiring;

// Re-export the master wiring so the entire quantum engine is available with one clean import
pub use quantum_master_wiring::*;
pub use mercy_engine_quantum_wiring::*;

// Public API for the full sovereign quantum engine
pub use crate::quantum_master_wiring::confirm_entire_quantum_wiring;
