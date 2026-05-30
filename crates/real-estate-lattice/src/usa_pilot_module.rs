//! USA Pilot Module — RREL v14.3
//! Now integrated with AttomDataProvider + caching

use crate::usa_attom_data_provider::{AttomDataProvider, UsaDataProvider};
use crate::usa_regulatory_engine::UsaRegulatoryEngine;
use crate::usa_state_adapters::{UsaStateAdapters, UsState};
use crate::RREL_VERSION;
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaPilotReport { /* ... existing fields ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaOfferFlowReport {
    pub state: String,
    pub passed_regulatory: bool,
    pub mercy_valence: f64,
    pub quantum_consensus: f64,
    pub federal_issues: Vec<String>,
    pub state_issues: Vec<String>,
    pub summary: String,
    // New enriched fields from data provider
    pub external_property_profile: Option<crate::usa_attom_cache::PropertyProfile>,
    pub external_risk_signals: Option<crate::usa_attom_cache::RiskSignals>,
}

pub struct UsaPilotModule {
    state_adapters: UsaStateAdapters,
    regulatory_engine: UsaRegulatoryEngine,
    data_provider: AttomDataProvider,
}

impl UsaPilotModule {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        Self {
            state_adapters: UsaStateAdapters::new(
                mercy_engine.clone(),
                quantum_swarm.clone(),
                world_governance.clone(),
            ),
            regulatory_engine: UsaRegulatoryEngine::new(
                mercy_engine,
                quantum_swarm,
                world_governance,
            ),
            data_provider: AttomDataProvider::new(),
        }
    }

    pub async fn process_usa_offer_flow(
        &mut self,
        state: UsState,
        transaction_details: &str,
        price: f64,
        game: &mut PowrushGame,
        property_identifier: Option<&str>, // e.g. address or parcel ID
    ) -> Result<UsaOfferFlowReport, crate::RrelError> {
        info!("🇺🇸 Processing USA offer flow for {:?}", state);

        let regulatory_result = self.regulatory_engine
            .check_usa_transaction(&format!("{:?}", state), transaction_details, price, game)
            .await
            .map_err(|e| crate::RrelError::Other(format!("Regulatory check failed: {}", e)))?;

        // Enrich with external data provider (cached)
        let mut external_profile = None;
        let mut external_risk = None;

        if let Some(identifier) = property_identifier {
            if let Ok(profile) = self.data_provider.get_property_profile(state, identifier).await {
                external_profile = Some(profile);
            }
            if let Ok(signals) = self.data_provider.get_risk_signals(state, identifier).await {
                external_risk = Some(signals);
            }
        }

        let summary = if regulatory_result.passed {
            format!("USA offer cleared regulatory checks in {:?}", state)
        } else {
            format!("USA offer has regulatory issues in {:?}", state)
        };

        let report = UsaOfferFlowReport {
            state: format!("{:?}", state),
            passed_regulatory: regulatory_result.passed,
            mercy_valence: regulatory_result.mercy_valence,
            quantum_consensus: regulatory_result.quantum_consensus,
            federal_issues: regulatory_result.federal_issues,
            state_issues: regulatory_result.state_issues,
            summary,
            external_property_profile: external_profile,
            external_risk_signals: external_risk,
        };

        Ok(report)
    }
}
