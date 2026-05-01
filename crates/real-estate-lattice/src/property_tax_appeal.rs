//! Property Tax Appeal & Optimization Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Property Tax Appeal System
//!
//! Helps owners reduce property tax burden through mercy-first, data-driven appeals.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum PropertyTaxAppealError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyTaxAppealRequest {
    pub property_mls_id: String,
    pub current_assessed_value: f64,
    pub proposed_assessed_value: f64,
    pub reason: String,                    // e.g. "Comparable sales", "Property condition", "Market decline"
    pub owner_cehi: f64,
    pub years_owned: u32,
}

pub struct PropertyTaxAppealEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl PropertyTaxAppealEngine {
    pub fn new(
        mercy_engine: MercyEngine,
        quantum_swarm: QuantumSwarmOrchestrator,
        world_governance: WorldGovernanceEngine,
    ) -> Self {
        Self {
            mercy_engine,
            quantum_swarm,
            world_governance,
        }
    }

    pub async fn process_tax_appeal(
        &mut self,
        request: &PropertyTaxAppealRequest,
        game: &mut PowrushGame,
    ) -> Result<String, PropertyTaxAppealError> {
        info!("📉 Processing property tax appeal for {} (RREL v{})", request.property_mls_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve tax appeal for {}", request.property_mls_id),
                "Property Tax Appeal",
                4.8,
                0.97,
            )
            .await
            .map_err(|_| PropertyTaxAppealError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(PropertyTaxAppealError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve property tax appeal", 0.82)
            .await
            .map_err(|_| PropertyTaxAppealError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(PropertyTaxAppealError::QuantumConsensusTooLow(consensus));
        }

        let potential_savings = (request.current_assessed_value - request.proposed_assessed_value) * 0.012; // \~1.2% effective tax rate

        // Trigger positive world impact
        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::USA_TexasPropertyTaxAppealGenerated, game)
            .await;

        let result = format!(
            "✅ PROPERTY TAX APPEAL APPROVED (RREL v{})\n\
             Property: {}\n\
             Current Assessed: ${:.0}\n\
             Proposed Assessed: ${:.0}\n\
             Estimated Annual Savings: ${:.0}\n\
             Reason: {}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: APPEAL FILED — STRONG CASE",
            RREL_VERSION,
            request.property_mls_id,
            request.current_assessed_value,
            request.proposed_assessed_value,
            potential_savings,
            request.reason,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
