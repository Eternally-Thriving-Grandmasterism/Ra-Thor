//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! USA MLS Adapter — RREL v0.6.0
//! Generic + State-Specific MLS Integration Layer for All 50 States
//! Mercy-Gated • Quantum Swarm • Immutable Legal Lattice
//!
//! Derived from RREL-USA-Expansion-Codex-v0.6.0.md

use crate::RREL_VERSION;
use crate::usa_regulatory_engine::{UsaRegulatoryEngine, UsaRegulatoryResult};
use crate::rrel_compliance::AiImageRecord;
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
    #[error("No real MLS connection configured for state {0}")]
    NotConfigured(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaListing {
    pub mls_id: String,
    pub state: String,
    /// Price as reported by the MLS data source. None when not yet sourced.
    pub price: Option<f64>,
    pub description: String,
    /// Images with alteration labels per RECO AI guidance.
    pub photos: Vec<AiImageRecord>,
    pub address: String,
    /// Identifier of the verified data source (e.g. MLS feed name + record ID).
    pub source: Option<String>,
    /// True only after a registrant has cross-checked against verifiable property documents.
    pub registrant_verified: bool,
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

    /// Fetch new listings for any USA state.
    ///
    /// Returns `Err(NotConfigured)` when no real MLS connection is available.
    /// A real implementation would call state-specific MLS APIs (CRMLS, FMLS, HAR,
    /// OneKey, etc.) with proper auth and return listings with verified sources.
    ///
    /// No price is emitted until a sourced, registrant-verified listing is available.
    pub async fn fetch_new_listings(&self, state: &str) -> Result<Vec<UsaListing>, UsaMlsError> {
        info!(
            "🇺🇸 Fetching new MLS listings for {} (RREL v{}) — no live MLS connection",
            state, RREL_VERSION
        );

        // No real MLS connection is wired. Return NotConfigured so callers know
        // that no data (not invented data) is available. Wire a real MLS feed here.
        Err(UsaMlsError::NotConfigured(state.to_string()))
    }

    /// Full end-to-end processing for a USA listing (mercy + quantum + regulatory)
    pub async fn process_usa_listing(
        &mut self,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, UsaMlsError> {
        // Require a sourced listing — no invented prices processed downstream
        if listing.source.is_none() {
            return Err(UsaMlsError::MlsFetchFailed {
                state: listing.state.clone(),
                message: "Listing has no verified source. A cited MLS source is \
                          required before regulatory processing.".to_string(),
            });
        }

        let price_for_check = listing.price.unwrap_or(0.0);

        let result = self.regulatory_engine
            .check_usa_transaction(
                &listing.state,
                &listing.description,
                price_for_check,
                game,
            )
            .await?;

        if result.passed {
            info!(
                "✅ USA listing {} approved in {} — mercy {:.2}, consensus {:.2}",
                listing.mls_id, listing.state, result.mercy_valence, result.quantum_consensus
            );
        } else {
            info!("❌ USA listing {} blocked — regulatory issues detected", listing.mls_id);
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Test-only mock data — clearly scoped to #[cfg(test)]
// ---------------------------------------------------------------------------

#[cfg(test)]
pub mod test_fixtures {
    use super::*;

    /// Returns a clearly-labelled mock listing for unit tests only.
    /// This data is synthetic and must never be presented as real property information.
    pub fn mock_listing(state: &str) -> UsaListing {
        UsaListing {
            mls_id: format!("{}-MOCK-001", state),
            state: state.to_string(),
            price: None, // price requires a cited source
            description: "[MOCK — synthetic test data, not a real property]".to_string(),
            photos: vec![],
            address: "[MOCK — no real address]".to_string(),
            source: Some("test-fixture".to_string()),
            registrant_verified: false,
        }
    }
}
