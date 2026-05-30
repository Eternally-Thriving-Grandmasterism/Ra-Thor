//! USA Pilot Module — RREL v14.3 Structured
//! Implements real USA offer processing logic using the regulatory engine

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
pub struct UsaPilotReport {
    pub listings_processed: u32,
    pub average_mercy_valence: f64,
    pub average_quantum_consensus: f64,
    pub regulatory_issues_prevented: u32,
    pub states_covered: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsaOfferFlowReport {
    pub state: String,
    pub passed_regulatory: bool,
    pub mercy_valence: f64,
    pub quantum_consensus: f64,
    pub federal_issues: Vec<String>,
    pub state_issues: Vec<String>,
    pub summary: String,
}

pub struct UsaPilotModule {
    state_adapters: UsaStateAdapters,
    regulatory_engine: UsaRegulatoryEngine,
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
        }
    }

    pub async fn process_usa_listings(
        &mut self,
        states: &[UsState],
        game: &mut PowrushGame,
    ) -> Result<UsaPilotReport, crate::RrelError> {
        info!("🇺🇸 RREL USA Pilot (v{}) — Processing {} states", RREL_VERSION, states.len());

        let mut total_processed = 0;
        let mut total_mercy = 0.0;
        let mut total_consensus = 0.0;
        let mut issues_prevented = 0;
        let mut states_covered = Vec::new();

        for state in states {
            states_covered.push(format!("{:?}", state));
            let listings = self.state_adapters.fetch_and_validate_state_listings(*state).await?;

            for listing in listings {
                let result = self.state_adapters.process_state_listing(*state, &listing, game).await?;
                if result.passed {
                    total_processed += 1;
                    total_mercy += result.mercy_valence;
                    total_consensus += result.quantum_consensus;
                } else {
                    issues_prevented += 1;
                }
            }
        }

        let report = UsaPilotReport {
            listings_processed: total_processed,
            average_mercy_valence: if total_processed > 0 { total_mercy / total_processed as f64 } else { 0.0 },
            average_quantum_consensus: if total_processed > 0 { total_consensus / total_processed as f64 } else { 0.0 },
            regulatory_issues_prevented: issues_prevented,
            states_covered,
            timestamp: chrono::Utc::now(),
        };

        Ok(report)
    }

    /// Implemented USA offer processing logic (state-aware + regulatory engine)
    pub async fn process_usa_offer_flow(
        &mut self,
        state: UsState,
        transaction_details: &str,
        price: f64,
        game: &mut PowrushGame,
    ) -> Result<UsaOfferFlowReport, crate::RrelError> {
        info!("🇺🇸 Processing USA offer flow for {:?}", state);

        let regulatory_result = self.regulatory_engine
            .check_usa_transaction(&format!("{:?}", state), transaction_details, price, game)
            .await
            .map_err(|e| crate::RrelError::Other(format!("Regulatory check failed: {}", e)))?;

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
        };

        Ok(report)
    }
}
