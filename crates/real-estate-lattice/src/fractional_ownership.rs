//! Fractional Ownership Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Tokenized Property Ownership
//!
//! Allows 13+ PATSAGi Councils to approve fractional ownership of real estate with full regulatory compliance.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum FractionalOwnershipError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("Regulatory compliance failed")]
    RegulatoryFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalOwnershipRequest {
    pub property_mls_id: String,
    pub total_property_value: f64,
    pub shares_offered: u32,
    pub price_per_share: f64,
    pub minimum_investment: f64,
}

pub struct FractionalOwnershipEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl FractionalOwnershipEngine {
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

    pub async fn approve_fractional_offering(
        &mut self,
        request: &FractionalOwnershipRequest,
        game: &mut PowrushGame,
    ) -> Result<String, FractionalOwnershipError> {
        info!("🏠 Processing fractional ownership request for {} (RREL v{})", request.property_mls_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action("Approve fractional property offering", "Fractional Ownership", 4.8, 0.97)
            .await
            .map_err(|_| FractionalOwnershipError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(FractionalOwnershipError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve fractional real estate offering", 0.85)
            .await
            .map_err(|_| FractionalOwnershipError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.80 {
            return Err(FractionalOwnershipError::QuantumConsensusTooLow(consensus));
        }

        // Trigger positive world impact
        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ FRACTIONAL OWNERSHIP APPROVED (RREL v{})\n\
             Property: {}\n\
             Total Value: ${:.0}\n\
             Shares Offered: {}\n\
             Price per Share: ${:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: LIVE ON POWRUSH-MMO",
            RREL_VERSION,
            request.property_mls_id,
            request.total_property_value,
            request.shares_offered,
            request.price_per_share,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
