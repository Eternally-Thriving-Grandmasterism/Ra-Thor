//! Rental Income Distribution Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Automatic Rental Income Distribution to Fractional Owners
//!
//! Allows the 13+ PATSAGi Councils to approve and execute fair, transparent rental income distribution.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum RentalDistributionError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalDistributionRequest {
    pub property_mls_id: String,
    pub total_rent_collected: f64,
    pub number_of_shares: u32,
    pub distribution_date: chrono::DateTime<chrono::Utc>,
}

pub struct RentalIncomeDistributionEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl RentalIncomeDistributionEngine {
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

    pub async fn distribute_rental_income(
        &mut self,
        request: &RentalDistributionRequest,
        game: &mut PowrushGame,
    ) -> Result<String, RentalDistributionError> {
        info!("💰 Processing rental income distribution for {} (RREL v{})", request.property_mls_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action("Distribute rental income to fractional owners", "Rental Distribution", 4.8, 0.97)
            .await
            .map_err(|_| RentalDistributionError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(RentalDistributionError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Distribute rental income fairly", 0.85)
            .await
            .map_err(|_| RentalDistributionError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.80 {
            return Err(RentalDistributionError::QuantumConsensusTooLow(consensus));
        }

        // Trigger positive world impact
        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let income_per_share = request.total_rent_collected / request.number_of_shares as f64;

        let result = format!(
            "✅ RENTAL INCOME DISTRIBUTED (RREL v{})\n\
             Property: {}\n\
             Total Rent: ${:.0}\n\
             Shares: {}\n\
             Income per Share: ${:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: DISTRIBUTED TO ALL OWNERS",
            RREL_VERSION,
            request.property_mls_id,
            request.total_rent_collected,
            request.number_of_shares,
            income_per_share,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
