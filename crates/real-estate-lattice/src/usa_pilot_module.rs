//! USA Pilot Module — RREL v0.5.21
//! Central orchestration layer using the unified 50-state system
//! Mercy-gated • Quantum Swarm • 13+ PATSAGi Councils

use crate::RREL_VERSION;
use crate::usa_state_adapters::{UsaStateAdapters, UsState};
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

pub struct UsaPilotModule {
    state_adapters: UsaStateAdapters,
}

impl UsaPilotModule {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        Self {
            state_adapters: UsaStateAdapters::new(mercy_engine, quantum_swarm, world_governance),
        }
    }

    pub async fn process_usa_listings(
        &mut self,
        states: &[UsState],
        game: &mut PowrushGame,
    ) -> Result<UsaPilotReport, crate::RrelError> {
        info!("🇺🇸 RREL USA Pilot (v{}) — Processing {} states using unified adapter", RREL_VERSION, states.len());

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

        info!("✅ USA Pilot Report: {} listings processed across {} states | Issues prevented: {}", total_processed, states.len(), issues_prevented);
        Ok(report)
    }
}
