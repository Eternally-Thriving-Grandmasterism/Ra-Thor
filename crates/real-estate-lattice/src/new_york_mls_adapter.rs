//! New York MLS Adapter — RREL v0.5.21
//! State-Specific Adapter for New York (OneKey MLS, REBNY, Hudson Gateway)
//! Mercy-Gated • Quantum Swarm • Rent Stabilization + Co-op Board + Lead Paint Disclosures
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
pub enum NewYorkMlsError {
    #[error("New York rent stabilization status not verified for listing {0}")]
    RentStabilizationNotVerified(String),
    #[error("Co-op board approval tracking or lead paint disclosure missing for listing {0}")]
    CoopOrLeadIssue(String),
    #[error("New York regulatory check failed: {0}")]
    RegulatoryFailed(#[from] crate::usa_regulatory_engine::UsaRegulatoryError),
}

pub struct NewYorkMlsAdapter {
    base_adapter: UsaMlsAdapter,
}

impl NewYorkMlsAdapter {
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

    /// Fetch new New York listings with extra NY-specific validation
    pub async fn fetch_new_york_listings(&self) -> Result<Vec<UsaListing>, NewYorkMlsError> {
        info!("🇺🇸🇳🇾 Fetching new New York MLS listings (RREL v{})", RREL_VERSION);

        let listings = self.base_adapter.fetch_new_listings("NY").await?;

        let mut validated = Vec::new();
        for listing in listings {
            if listing.description.to_lowercase().contains("rent stabilization") 
                && !listing.description.to_lowercase().contains("verified") {
                return Err(NewYorkMlsError::RentStabilizationNotVerified(listing.mls_id.clone()));
            }

            if listing.description.to_lowercase().contains("co-op") 
                || listing.description.to_lowercase().contains("lead paint") {
                if !listing.description.to_lowercase().contains("board") 
                    && !listing.description.to_lowercase().contains("disclosure") {
                    return Err(NewYorkMlsError::CoopOrLeadIssue(listing.mls_id.clone()));
                }
            }

            validated.push(listing);
        }

        Ok(validated)
    }

    /// Full New York processing pipeline
    pub async fn process_new_york_listing(
        &mut self,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, NewYorkMlsError> {
        let result = self.base_adapter
            .process_usa_listing(listing, game)
            .await?;

        if !result.passed {
            return Ok(result);
        }

        if listing.description.to_lowercase().contains("rent stabilization") 
            && !listing.description.to_lowercase().contains("verified") {
            return Err(NewYorkMlsError::RentStabilizationNotVerified(listing.mls_id.clone()));
        }

        info!("✅ New York listing {} fully approved — mercy {:.2}, consensus {:.2}",
              listing.mls_id, result.mercy_valence, result.quantum_consensus);

        Ok(result)
    }
}
