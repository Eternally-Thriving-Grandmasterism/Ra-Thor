//! Rental Income Distribution Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Fair & Ethical Rental Income Distribution
//!
//! Automatically distributes rental income to owners, fractional stakeholders, and tenants with full mercy-first logic.

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
    pub distribution_id: String,
    pub property_mls_id: String,
    pub monthly_rent_collected: f64,
    pub total_ownership_shares: f64,
    pub investor_cehi: f64,
    pub investor_share_percentage: f64,
    pub months_since_last_distribution: u8,
    pub tenant_relief_percentage: f64, // 0.0 - 0.15 (optional mercy-based tenant support)
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
        info!("💰 Processing rental income distribution {} (RREL v{})", request.distribution_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Distribute rental income for {}", request.property_mls_id),
                "Rental Income Distribution",
                request.investor_cehi,
                0.93,
            )
            .await
            .map_err(|_| RentalDistributionError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.87 {
            return Err(RentalDistributionError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve rental income distribution", 0.79)
            .await
            .map_err(|_| RentalDistributionError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.75 {
            return Err(RentalDistributionError::QuantumConsensusTooLow(consensus));
        }

        let investor_payout = request.monthly_rent_collected * request.investor_share_percentage;
        let platform_fee = investor_payout * 0.03;
        let tenant_relief = if request.tenant_relief_percentage > 0.0 {
            investor_payout * request.tenant_relief_percentage
        } else {
            0.0
        };
        let net_payout = investor_payout - platform_fee - tenant_relief;

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ RENTAL INCOME DISTRIBUTED (RREL v{})\n\
             Distribution ID: {}\n\
             Property: {}\n\
             Monthly Rent Collected: ${:.0}\n\
             Investor Share: {:.1}%\n\
             Gross Payout: ${:.0}\n\
             Platform Fee (3%): ${:.0}\n\
             Tenant Relief (Mercy): ${:.0}\n\
             Net Payout to Investor: ${:.0}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: MERCY-ALIGNED • FAIRLY DISTRIBUTED",
            RREL_VERSION,
            request.distribution_id,
            request.property_mls_id,
            request.monthly_rent_collected,
            request.investor_share_percentage * 100.0,
            investor_payout,
            platform_fee,
            tenant_relief,
            net_payout,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
