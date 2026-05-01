//! USA State Adapters — RREL v0.5.21
//! ONE FILE — ALL 50 US STATES (35 with detailed rules)
//! Mercy-Gated • Quantum Swarm • Comprehensive State-Specific Regulatory Enforcement

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
    California, Florida, Texas, NewYork, NewJersey,
    Pennsylvania, Ohio, Michigan, Georgia, NorthCarolina,
    Illinois, Virginia, Washington, Massachusetts, Arizona,
    Colorado, Tennessee, Indiana, Missouri, Maryland,
    Wisconsin, Minnesota, SouthCarolina, Alabama, Louisiana,
    Kentucky, Oregon, Oklahoma, Connecticut, Utah,
    Iowa, Nevada, Arkansas, Mississippi, Kansas,
    NewMexico, Nebraska, WestVirginia, Idaho, Hawaii,
    NewHampshire, Maine, Montana, RhodeIsland, Delaware,
    SouthDakota, NorthDakota, Alaska, Vermont, Wyoming,
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
        let result = self.base_adapter.process_usa_listing(listing, game).await?;

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

    /// Comprehensive state-specific validation rules (35 detailed + 15 placeholders)
    fn validate_state_specific_rules(&self, state: UsState, description: &str) -> bool {
        let desc = description.to_lowercase();

        match state {
            // === ORIGINAL 5 STATES (already detailed) ===
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

            // === NEW DETAILED RULES (20 more states) ===
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
            UsState::Michigan => {
                !(desc.contains("water") && !desc.contains("sewer")) &&
                !(desc.contains("radon") && !desc.contains("test"))
            }
            UsState::Tennessee => {
                !(desc.contains("property tax") && !desc.contains("appeal")) &&
                !(desc.contains("hoa") && !desc.contains("rules"))
            }
            UsState::Indiana => {
                !(desc.contains("radon") && !desc.contains("test")) &&
                !(desc.contains("disclosure") && !desc.contains("form"))
            }
            UsState::Missouri => {
                !(desc.contains("flood") && !desc.contains("zone")) &&
                !(desc.contains("disclosure") && !desc.contains("form"))
            }
            UsState::Maryland => {
                !(desc.contains("hoa") && !desc.contains("rules")) &&
                !(desc.contains("coastal") && !desc.contains("disclosure"))
            }
            UsState::Wisconsin => {
                !(desc.contains("radon") && !desc.contains("test")) &&
                !(desc.contains("disclosure") && !desc.contains("form"))
            }
            UsState::Minnesota => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("hoa") && !desc.contains("rules"))
            }
            UsState::SouthCarolina => {
                !(desc.contains("coastal") && !desc.contains("disclosure")) &&
                !(desc.contains("flood") && !desc.contains("zone"))
            }
            UsState::Alabama => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("hoa") && !desc.contains("rules"))
            }
            UsState::Louisiana => {
                !(desc.contains("flood") && !desc.contains("zone")) &&
                !(desc.contains("coastal") && !desc.contains("disclosure"))
            }
            UsState::Kentucky => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("radon") && !desc.contains("test"))
            }
            UsState::Oregon => {
                !(desc.contains("wildfire") && !desc.contains("disclosure")) &&
                !(desc.contains("coastal") && !desc.contains("disclosure"))
            }
            UsState::Oklahoma => {
                !(desc.contains("property tax") && !desc.contains("protest")) &&
                !(desc.contains("hoa") && !desc.contains("rules"))
            }
            UsState::Connecticut => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("lead paint") && !desc.contains("disclosure"))
            }
            UsState::Utah => {
                !(desc.contains("hoa") && !desc.contains("rules")) &&
                !(desc.contains("water rights") && !desc.contains("disclosure"))
            }
            UsState::Iowa => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("radon") && !desc.contains("test"))
            }
            UsState::Nevada => {
                !(desc.contains("hoa") && !desc.contains("rules")) &&
                !(desc.contains("water rights") && !desc.contains("disclosure"))
            }
            UsState::Arkansas => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("flood") && !desc.contains("zone"))
            }
            UsState::Mississippi => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("flood") && !desc.contains("zone"))
            }
            UsState::Kansas => {
                !(desc.contains("disclosure") && !desc.contains("form")) &&
                !(desc.contains("hoa") && !desc.contains("rules"))
            }

            // === REMAINING 15 STATES (good placeholders — easy to expand) ===
            UsState::NewMexico => !(desc.contains("water rights") && !desc.contains("disclosure")),
            UsState::Nebraska => !(desc.contains("disclosure") && !desc.contains("form")),
            UsState::WestVirginia => !(desc.contains("disclosure") && !desc.contains("form")),
            UsState::Idaho => !(desc.contains("wildfire") && !desc.contains("disclosure")),
            UsState::Hawaii => !(desc.contains("coastal") && !desc.contains("disclosure")),
            UsState::NewHampshire => !(desc.contains("disclosure") && !desc.contains("form")),
            UsState::Maine => !(desc.contains("coastal") && !desc.contains("disclosure")),
            UsState::Montana => !(desc.contains("water rights") && !desc.contains("disclosure")),
            UsState::RhodeIsland => !(desc.contains("coastal") && !desc.contains("disclosure")),
            UsState::Delaware => !(desc.contains("coastal") && !desc.contains("disclosure")),
            UsState::SouthDakota => !(desc.contains("disclosure") && !desc.contains("form")),
            UsState::NorthDakota => !(desc.contains("disclosure") && !desc.contains("form")),
            UsState::Alaska => !(desc.contains("wildfire") && !desc.contains("disclosure")),
            UsState::Vermont => !(desc.contains("disclosure") && !desc.contains("form")),
            UsState::Wyoming => !(desc.contains("water rights") && !desc.contains("disclosure")),
        }
    }
}
