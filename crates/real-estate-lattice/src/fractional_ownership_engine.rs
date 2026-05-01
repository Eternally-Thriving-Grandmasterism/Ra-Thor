//! Fractional Ownership Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Tokenized Real Estate Ownership
//!
//! Enables ethical fractional ownership with CEHI-weighted investor scoring and mercy-first governance.

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalOwnershipRequest {
    pub offering_id: String,
    pub property_mls_id: String,
    pub total_property_value: f64,
    pub fractional_share_percentage: f64, // e.g. 0.05 = 5%
    pub investor_cehi: f64,
    pub investor_years_in_market: u8,
    pub requested_investment_amount: f64,
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
        info!("🪙 Processing fractional ownership offering {} (RREL v{})", request.offering_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve fractional ownership for {}", request.property_mls_id),
                "Fractional Ownership",
                request.investor_cehi,
                0.94,
            )
            .await
            .map_err(|_| FractionalOwnershipError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.88 {
            return Err(FractionalOwnershipError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve fractional ownership terms", 0.80)
            .await
            .map_err(|_| FractionalOwnershipError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.76 {
            return Err(FractionalOwnershipError::QuantumConsensusTooLow(consensus));
        }

        let share_value = request.total_property_value * request.fractional_share_percentage;
        let expected_annual_yield = share_value * 0.065; // \~6.5% net yield assumption

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ FRACTIONAL OWNERSHIP APPROVED (RREL v{})\n\
             Offering ID: {}\n\
             Property: {}\n\
             Total Property Value: ${:.0}\n\
             Fractional Share: {:.1}%\n\
             Investment Amount: ${:.0}\n\
             Expected Annual Yield: ${:.0}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: MERCY-ALIGNED • TOKENIZED OWNERSHIP ENABLED",
            RREL_VERSION,
            request.offering_id,
            request.property_mls_id,
            request.total_property_value,
            request.fractional_share_percentage * 100.0,
            request.requested_investment_amount,
            expected_annual_yield,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
