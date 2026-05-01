//! California MLS Adapter — RREL v0.6.0
//! State-Specific Adapter for California (CRMLS, MLSListings, MetroList)
//! Mercy-Gated • Quantum Swarm • Wildfire + Rent Control + AB 1482 Compliance
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
pub enum CaliforniaMlsError {
    #[error("California wildfire disclosure missing for listing {0}")]
    WildfireDisclosureMissing(String),
    #[error("AB 1482 rent control compliance not verified for listing {0}")]
    RentControlNotVerified(String),
    #[error("California regulatory check failed: {0}")]
    RegulatoryFailed(#[from] crate::usa_regulatory_engine::UsaRegulatoryError),
}

pub struct CaliforniaMlsAdapter {
    base_adapter: UsaMlsAdapter,
}

impl CaliforniaMlsAdapter {
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

    /// Fetch new California listings with extra CA-specific validation
    pub async fn fetch_california_listings(&self) -> Result<Vec<UsaListing>, CaliforniaMlsError> {
        info!("🇺🇸🇨🇦 Fetching new California MLS listings (RREL v{})", RREL_VERSION);

        let listings = self.base_adapter.fetch_new_listings("CA").await?;

        // Extra California-specific pre-filtering
        let mut validated = Vec::new();
        for listing in listings {
            if listing.description.to_lowercase().contains("wildfire") 
                && !listing.description.to_lowercase().contains("disclosure") {
                return Err(CaliforniaMlsError::WildfireDisclosureMissing(listing.mls_id.clone()));
            }

            if listing.description.to_lowercase().contains("rent control") 
                && !listing.description.to_lowercase().contains("ab 1482") {
                return Err(CaliforniaMlsError::RentControlNotVerified(listing.mls_id.clone()));
            }

            validated.push(listing);
        }

        Ok(validated)
    }

    /// Full California processing pipeline (mercy + quantum + CA-specific rules)
    pub async fn process_california_listing(
        &mut self,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, CaliforniaMlsError> {
        // Run base USA regulatory check first
        let result = self.base_adapter
            .process_usa_listing(listing, game)
            .await?;

        if !result.passed {
            return Ok(result);
        }

        // Additional California-specific checks
        if listing.description.to_lowercase().contains("wildfire") 
            && !listing.description.to_lowercase().contains("disclosure") {
            return Err(CaliforniaMlsError::WildfireDisclosureMissing(listing.mls_id.clone()));
        }

        info!("✅ California listing {} fully approved — mercy {:.2}, consensus {:.2}",
              listing.mls_id, result.mercy_valence, result.quantum_consensus);

        Ok(result)
    }
}
