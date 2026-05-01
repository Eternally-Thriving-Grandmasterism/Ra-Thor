//! Texas MLS Adapter — RREL v0.5.21
//! State-Specific Adapter for Texas (HAR, NTREIS, SABOR)
//! Mercy-Gated • Quantum Swarm • Property Tax Protest + Homestead Exemption + Energy Disclosures
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
pub enum TexasMlsError {
    #[error("Texas property tax protest opportunity not documented for listing {0}")]
    PropertyTaxProtestMissing(String),
    #[error("Homestead exemption or energy-efficient disclosure missing for listing {0}")]
    HomesteadOrEnergyIssue(String),
    #[error("Texas regulatory check failed: {0}")]
    RegulatoryFailed(#[from] crate::usa_regulatory_engine::UsaRegulatoryError),
}

pub struct TexasMlsAdapter {
    base_adapter: UsaMlsAdapter,
}

impl TexasMlsAdapter {
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

    /// Fetch new Texas listings with extra TX-specific validation
    pub async fn fetch_texas_listings(&self) -> Result<Vec<UsaListing>, TexasMlsError> {
        info!("🇺🇸🇹🇽 Fetching new Texas MLS listings (RREL v{})", RREL_VERSION);

        let listings = self.base_adapter.fetch_new_listings("TX").await?;

        let mut validated = Vec::new();
        for listing in listings {
            if listing.description.to_lowercase().contains("property tax") 
                && !listing.description.to_lowercase().contains("protest") {
                return Err(TexasMlsError::PropertyTaxProtestMissing(listing.mls_id.clone()));
            }

            if listing.description.to_lowercase().contains("homestead") 
                || listing.description.to_lowercase().contains("energy") {
                if !listing.description.to_lowercase().contains("exemption") 
                    && !listing.description.to_lowercase().contains("efficient") {
                    return Err(TexasMlsError::HomesteadOrEnergyIssue(listing.mls_id.clone()));
                }
            }

            validated.push(listing);
        }

        Ok(validated)
    }

    /// Full Texas processing pipeline
    pub async fn process_texas_listing(
        &mut self,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, TexasMlsError> {
        let result = self.base_adapter
            .process_usa_listing(listing, game)
            .await?;

        if !result.passed {
            return Ok(result);
        }

        if listing.description.to_lowercase().contains("property tax") 
            && !listing.description.to_lowercase().contains("protest") {
            return Err(TexasMlsError::PropertyTaxProtestMissing(listing.mls_id.clone()));
        }

        info!("✅ Texas listing {} fully approved — mercy {:.2}, consensus {:.2}",
              listing.mls_id, result.mercy_valence, result.quantum_consensus);

        Ok(result)
    }
}
