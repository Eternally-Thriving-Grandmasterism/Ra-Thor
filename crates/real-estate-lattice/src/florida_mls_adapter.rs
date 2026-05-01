//! Florida MLS Adapter — RREL v0.5.21
//! State-Specific Adapter for Florida (FMLS, Stellar MLS, My Florida Regional)
//! Mercy-Gated • Quantum Swarm • Flood Zone + Hurricane + HOA/Condo Disclosures
//!
//! Derived from RREL-USA-Expansion-Codex-v0.6.0.md

use crate::RREL_VERSION;
use crate::usa_mls_adapter::{UsaListing, UsaMlsAdapter};
use crate::usa_regulatory_engine::UsaRegulatoryResult;
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum FloridaMlsError {
    #[error("Florida flood zone disclosure missing for listing {0}")]
    FloodZoneDisclosureMissing(String),
    #[error("Hurricane insurance / HOA financial health not verified for listing {0}")]
    HurricaneOrHoaIssue(String),
    #[error("Florida regulatory check failed: {0}")]
    RegulatoryFailed(#[from] crate::usa_regulatory_engine::UsaRegulatoryError),
}

pub struct FloridaMlsAdapter {
    base_adapter: UsaMlsAdapter,
}

impl FloridaMlsAdapter {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        let base_adapter = UsaMlsAdapter::new(
            mercy_engine,
            quantum_swarm,
            world_governance,
        );

        Self { base_adapter }
    }

    /// Fetch new Florida listings with extra FL-specific validation
    pub async fn fetch_florida_listings(&self) -> Result<Vec<UsaListing>, FloridaMlsError> {
        info!("🇺🇸🇫🇱 Fetching new Florida MLS listings (RREL v{})", RREL_VERSION);

        let listings = self.base_adapter.fetch_new_listings("FL").await?;

        let mut validated = Vec::new();
        for listing in listings {
            if listing.description.to_lowercase().contains("flood") 
                && !listing.description.to_lowercase().contains("zone") {
                return Err(FloridaMlsError::FloodZoneDisclosureMissing(listing.mls_id.clone()));
            }

            if listing.description.to_lowercase().contains("hurricane") 
                || listing.description.to_lowercase().contains("hoa") {
                if !listing.description.to_lowercase().contains("insurance") 
                    && !listing.description.to_lowercase().contains("financial") {
                    return Err(FloridaMlsError::HurricaneOrHoaIssue(listing.mls_id.clone()));
                }
            }

            validated.push(listing);
        }

        Ok(validated)
    }

    /// Full Florida processing pipeline
    pub async fn process_florida_listing(
        &mut self,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, FloridaMlsError> {
        let result = self.base_adapter
            .process_usa_listing(listing, game)
            .await?;

        if !result.passed {
            return Ok(result);
        }

        if listing.description.to_lowercase().contains("flood") 
            && !listing.description.to_lowercase().contains("zone") {
            return Err(FloridaMlsError::FloodZoneDisclosureMissing(listing.mls_id.clone()));
        }

        info!("✅ Florida listing {} fully approved — mercy {:.2}, consensus {:.2}",
              listing.mls_id, result.mercy_valence, result.quantum_consensus);

        Ok(result)
    }
}
