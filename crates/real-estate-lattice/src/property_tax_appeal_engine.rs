//! Property Tax Appeal Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Tax Appeal Optimization
//!
//! Helps owners appeal property taxes with full ethical oversight and maximum savings.

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
    pub appeal_id: String,
    pub property_mls_id: String,
    pub current_assessed_value: f64,
    pub proposed_assessed_value: f64,
    pub appeal_reason: String,
    pub owner_cehi: f64,
    pub years_owned: u16,
    pub previous_appeals_won: u8,
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
        info!("🏛️ Processing property tax appeal {} (RREL v{})", request.appeal_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve tax appeal for {}", request.property_mls_id),
                "Property Tax Appeal",
                request.owner_cehi,
                0.95,
            )
            .await
            .map_err(|_| PropertyTaxAppealError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(PropertyTaxAppealError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve property tax appeal", 0.81)
            .await
            .map_err(|_| PropertyTaxAppealError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.77 {
            return Err(PropertyTaxAppealError::QuantumConsensusTooLow(consensus));
        }

        let potential_savings = (request.current_assessed_value - request.proposed_assessed_value) * 0.012; // \~1.2% effective tax rate

        let recommended_action = if potential_savings > 2500.0 {
            "STRONG CASE — File appeal immediately with comparable sales evidence"
        } else if potential_savings > 800.0 {
            "GOOD CASE — File appeal with professional appraisal support"
        } else {
            "MODERATE CASE — Consider filing with strong documentation"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::USA_TexasPropertyTaxAppealGenerated, game)
            .await;

        let result = format!(
            "✅ PROPERTY TAX APPEAL PROCESSED (RREL v{})\n\
             Appeal ID: {}\n\
             Property: {}\n\
             Current Assessed: ${:.0}\n\
             Proposed Assessed: ${:.0}\n\
             Potential Annual Savings: ${:.0}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Recommended Action: {}\n\
             Status: MERCY-ALIGNED • QUANTUM-VERIFIED",
            RREL_VERSION,
            request.appeal_id,
            request.property_mls_id,
            request.current_assessed_value,
            request.proposed_assessed_value,
            potential_savings,
            mercy_valence,
            consensus,
            recommended_action
        );

        info!("{}", result);
        Ok(result)
    }
}
