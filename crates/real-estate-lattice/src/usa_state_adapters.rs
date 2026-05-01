//! USA State Adapters — RREL v0.5.21
//! ONE FILE — ALL 50 US STATES (15+ with detailed rules)
//! Mercy-Gated • Quantum Swarm • Comprehensive State-Specific Regulatory Enforcement
//!
//! This single elegant file handles the entire United States with easy extensibility.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UsState {
    California,
    Florida,
    Texas,
    NewYork,
    NewJersey,
    Pennsylvania,
    Ohio,
    Michigan,
    Georgia,
    NorthCarolina,
    Illinois,
    Virginia,
    Washington,
    Massachusetts,
    Arizona,
    Colorado,
    Tennessee,
    Indiana,
    Missouri,
    Maryland,
    Wisconsin,
    Minnesota,
    SouthCarolina,
    Alabama,
    Louisiana,
    Kentucky,
    Oregon,
    Oklahoma,
    Connecticut,
    Utah,
    Iowa,
    Nevada,
    Arkansas,
    Mississippi,
    Kansas,
    NewMexico,
    Nebraska,
    WestVirginia,
    Idaho,
    Hawaii,
    NewHampshire,
    Maine,
    Montana,
    RhodeIsland,
    Delaware,
    SouthDakota,
    NorthDakota,
    Alaska,
    Vermont,
    Wyoming,
}

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum UsaStateAdapterError {
    #[error("State-specific regulatory issue in {state:?}: {message}")]
    StateSpecificIssue { state: UsState, message: String },
    #[error("Base regulatory check failed: {0}")]
    BaseRegulatoryFailed(#[from] crate::usa_regulatory_engine::UsaRegulatoryError),
}

pub struct UsaStateAdapters {
    base_adapter: UsaMlsAdapter,
}

impl UsaStateAdapters {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        Self {
            base_adapter: UsaMlsAdapter::new(mercy_engine, quantum_swarm, world_governance),
        }
    }

    pub async fn fetch_and_validate_state_listings(
        &self,
        state: UsState,
    ) -> Result<Vec<UsaListing>, UsaStateAdapterError> {
        let state_str = format!("{:?}", state);
        let listings = self.base_adapter.fetch_new_listings(&state_str).await?;

        let mut validated = Vec::new();
        for listing in listings {
            if self.validate_state_specific_rules(state, &listing.description) {
                validated.push(listing);
            } else {
                return Err(UsaStateAdapterError::StateSpecificIssue {
                    state,
                    message: format!("State-specific rule violation in {}", state_str),
                });
            }
        }
        Ok(validated)
    }

    pub async fn process_state_listing(
        &mut self,
        state: UsState,
        listing: &UsaListing,
        game: &mut PowrushGame,
    ) -> Result<UsaRegulatoryResult, UsaStateAdapterError> {
        let result = self.base_adapter
            .process_usa_listing(listing, game)
            .await?;

        if !result.passed {
            return Ok(result);
        }

        if !self.validate_state_specific_rules(state, &listing.description) {
            return Err(UsaStateAdapterError::StateSpecificIssue {
                state,
                message: "State-specific regulatory rule failed".to_string(),
            });
        }

        info!("✅ {} listing {} fully approved — mercy {:.2}, consensus {:.2}",
              format!("{:?}", state), listing.mls_id, result.mercy_valence, result.quantum_consensus);

        Ok(result)
    }

    /// Comprehensive state-specific validation rules (easy to extend)
    fn validate_state_specific_rules(&self, state: UsState, description: &str) -> bool {
        let desc = description.to_lowercase();

        match state {
            UsState::California => {
                !(desc.contains("wildfire") && !desc.contains("disclosure")) &&
                !(desc.contains("rent control") && !desc.contains("ab 1482"))
            }
            UsState::Florida => {
                !(desc.contains("flood") && !desc.contains("zone")) &&
                !(desc.contains("hurricane") && !desc.contains("insurance")) &&
                !(desc.contains("hoa") && !desc.contains("financial"))
            }
            UsState::Texas => {
                !(desc.contains("property tax") && !desc.contains("protest")) &&
                !(desc.contains("homestead") && !desc.contains("exemption"))
            }
            UsState::NewYork => {
                !(desc.contains("rent stabilization") && !desc.contains("verified")) &&
                !(desc.contains("co-op") && !desc.contains("board")) &&
                !(desc.contains("lead paint") && !desc.contains("disclosure"))
            }
            UsState::NewJersey => {
                !(desc.contains("coastal") && !desc.contains("disclosure")) &&
                !(desc.contains("affordable housing") && !desc.contains("mount laurel"))
            }
            UsState::Pennsylvania => {
                !(desc.contains("radon") && !desc.contains("test")) &&
                !(desc.contains("disclosure") && !desc.contains("act 66"))
            }
            UsState::Illinois => {
                !(desc.contains("property tax") && !desc.contains("appeal")) &&
                !(desc.contains("chicago") && !desc.contains("rules"))
            }
            UsState::Ohio => {
                !(desc.contains("radon") && !desc.contains("test"))
            }
            UsState::Georgia => {
                !(desc.contains("property tax") && !desc.contains("reassessment")) &&
                !(desc.contains("flood") && !desc.contains("zone"))
            }
            UsState::Washington => {
                !(desc.contains("rent control") && !desc.contains("seattle")) &&
                !(desc.contains("coastal") && !desc.contains("disclosure"))
            }
            UsState::Massachusetts => {
                !(desc.contains("rent control") && !desc.contains("boston")) &&
                !(desc.contains("lead paint") && !desc.contains("disclosure"))
            }
            UsState::Arizona => {
                !(desc.contains("hoa") && !desc.contains("rules")) &&
                !(desc.contains("water rights") && !desc.contains("disclosure"))
            }
            UsState::Colorado => {
                !(desc.contains("wildfire") && !desc.contains("disclosure")) &&
                !(desc.contains("hoa") && !desc.contains("financial"))
            }
            UsState::Virginia => {
                !(desc.contains("hoa") && !desc.contains("rules")) &&
                !(desc.contains("flood") && !desc.contains("zone")) &&
                !(desc.contains("historic") && !desc.contains("district"))
            }
            UsState::NorthCarolina => {
                !(desc.contains("coastal") && !desc.contains("stormwater")) &&
                !(desc.contains("flood") && !desc.contains("zone"))
            }
            // All other states fall back to base federal rules only
            _ => true,
        }
    }
}
