//! USA MLS Adapter — RREL v0.6.0
//! Generic + State-Specific MLS Integration Layer for All 50 States
//! Mercy-Gated • Quantum Swarm • Immutable Legal Lattice
//!
//! Derived from RREL-USA-Expansion-Codex-v0.6.0.md

use crate::RREL_VERSION;
use crate::usa_regulatory_engine::{UsaRegulatoryEngine, UsaRegulatoryResult};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum UsaMlsError {
    #[error("MLS fetch failed for state {state}: {message}")]
    MlsFetchFailed { state: String, message: String },
    #[error("Regulatory check failed: {0}")]
    RegulatoryCheckFailed(#[from] crate::usa_regulatory_engine::UsaRegulatoryError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaListing {
    pub mls_id: String,
    pub state: String,
    pub price: f64,
    pub description: String,
    pub photos: Vec<String>,
    pub address: String,
}

pub struct UsaMlsAdapter {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    regulatory_engine: UsaRegulatoryEngine,
}

impl UsaMlsAdapter {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        let regulatory_engine = UsaRegulatoryEngine::new(
            mercy_engine.clone(),
            quantum_swarm.clone(),
            world_governance,
        );

        Self {
            mercy_engine,
            quantum_swarm,
            regulatory_engine,
        }
    }

    /// Fetch new listings for any USA state (extensible)
    pub async fn fetch_new_listings(&self, state: &str) -> Result<Vec<UsaListing>, UsaMlsError> {
        info!("🇺🇸 Fetching new MLS listings for {} (RREL v{})", state, RREL_VERSION);

        // Placeholder for real MLS API calls (CRMLS, FMLS, HAR, OneKey, etc.)
        // In production this would call state-specific MLS APIs with proper auth
        let mock_listings = vec![
            UsaListing {
                mls_id: format!("{}-2026-0429-001", state),
                state: state.to_string(),
                price: 875000.0,
                description: "Beautiful 4-bed home with modern kitchen and large backyard. TILA disclosure included. No kickbacks.".to_string(),
                photos: vec!["photo1.jpg".to_string(), "photo2.jpg".to_string()],
                address: "123 Main St, Example City, CA".to_string(),
            },
        ];

        Ok(mock_listings)
    }

    /// Full end-to-end processing for a USA listing (mercy + quantum + regulatory)
    pub async fn process_usa_listing(
        &mut self,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, UsaMlsError> {
        // Run full regulatory check (federal + state)
        let result = self.regulatory_engine
            .check_usa_transaction(
                &listing.state,
                &listing.description,
                listing.price,
                game,
            )
            .await?;

        if result.passed {
            info!("✅ USA listing {} approved in {} — mercy {:.2}, consensus {:.2}",
                  listing.mls_id, listing.state, result.mercy_valence, result.quantum_consensus);
        } else {
            info!("❌ USA listing {} blocked — regulatory issues detected", listing.mls_id);
        }

        Ok(result)
    }
}
