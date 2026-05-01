//! New Jersey MLS Adapter — RREL v0.5.21
//! State-Specific Adapter for New Jersey (NJMLS, Garden State MLS)
//! Mercy-Gated • Quantum Swarm • Coastal Zone + Flood + Mount Laurel Affordable Housing
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
pub enum NewJerseyMlsError {
    #[error("New Jersey coastal zone or flood disclosure missing for listing {0}")]
    CoastalOrFloodDisclosureMissing(String),
    #[error("Mount Laurel affordable housing set-aside not verified for listing {0}")]
    MountLaurelIssue(String),
    #[error("New Jersey regulatory check failed: {0}")]
    RegulatoryFailed(#[from] crate::usa_regulatory_engine::UsaRegulatoryError),
}

pub struct NewJerseyMlsAdapter {
    base_adapter: UsaMlsAdapter,
}

impl NewJerseyMlsAdapter {
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

    /// Fetch new New Jersey listings with extra NJ-specific validation
    pub async fn fetch_new_jersey_listings(&self) -> Result<Vec<UsaListing>, NewJerseyMlsError> {
        info!("🇺🇸🇳🇯 Fetching new New Jersey MLS listings (RREL v{})", RREL_VERSION);

        let listings = self.base_adapter.fetch_new_listings("NJ").await?;

        let mut validated = Vec::new();
        for listing in listings {
            if (listing.description.to_lowercase().contains("coastal") 
                || listing.description.to_lowercase().contains("flood"))
                && !listing.description.to_lowercase().contains("disclosure") {
                return Err(NewJerseyMlsError::CoastalOrFloodDisclosureMissing(listing.mls_id.clone()));
            }

            if listing.description.to_lowercase().contains("affordable housing") 
                || listing.description.to_lowercase().contains("set-aside") {
                if !listing.description.to_lowercase().contains("mount laurel") {
                    return Err(NewJerseyMlsError::MountLaurelIssue(listing.mls_id.clone()));
                }
            }

            validated.push(listing);
        }

        Ok(validated)
    }

    /// Full New Jersey processing pipeline
    pub async fn process_new_jersey_listing(
        &mut self,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, NewJerseyMlsError> {
        let result = self.base_adapter
            .process_usa_listing(listing, game)
            .await?;

        if !result.passed {
            return Ok(result);
        }

        if (listing.description.to_lowercase().contains("coastal") 
            || listing.description.to_lowercase().contains("flood"))
            && !listing.description.to_lowercase().contains("disclosure") {
            return Err(NewJerseyMlsError::CoastalOrFloodDisclosureMissing(listing.mls_id.clone()));
        }

        info!("✅ New Jersey listing {} fully approved — mercy {:.2}, consensus {:.2}",
              listing.mls_id, result.mercy_valence, result.quantum_consensus);

        Ok(result)
    }
}
