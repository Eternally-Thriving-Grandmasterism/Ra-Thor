//! Multi-Generational Wealth Transfer & Family Office Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Legacy Planning
//!
//! Plans and executes wealth transfer across generations with CEHI inheritance bonuses and mercy-first governance.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum WealthTransferError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WealthTransferRequest {
    pub transfer_id: String,
    pub property_mls_id: String,
    pub current_owner_cehi: f64,
    pub next_generation_cehi: f64,
    pub total_estate_value: f64,
    pub percentage_to_next_generation: f64,
    pub generations_ahead: u8, // 1, 2, or 3
    pub family_consensus_score: f64,
}

pub struct MultiGenerationalWealthTransferEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl MultiGenerationalWealthTransferEngine {
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

    pub async fn process_wealth_transfer(
        &mut self,
        request: &WealthTransferRequest,
        game: &mut PowrushGame,
    ) -> Result<String, WealthTransferError> {
        info!("🌳 Processing multi-generational wealth transfer {} (RREL v{})", request.transfer_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve wealth transfer for {}", request.property_mls_id),
                "Multi-Generational Wealth Transfer",
                (request.current_owner_cehi + request.next_generation_cehi) / 2.0,
                0.95,
            )
            .await
            .map_err(|_| WealthTransferError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(WealthTransferError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve multi-generational wealth transfer", 0.81)
            .await
            .map_err(|_| WealthTransferError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.77 {
            return Err(WealthTransferError::QuantumConsensusTooLow(consensus));
        }

        let transfer_amount = request.total_estate_value * request.percentage_to_next_generation;
        let epigenetic_bonus = if request.generations_ahead >= 3 { 0.12 } else if request.generations_ahead >= 2 { 0.08 } else { 0.04 };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ MULTI-GENERATIONAL WEALTH TRANSFER APPROVED (RREL v{})\n\
             Transfer ID: {}\n\
             Property: {}\n\
             Total Estate Value: ${:.0}\n\
             Transferred to Next Generation: ${:.0} ({:.1}%)\n\
             Generations Ahead: {}\n\
             Epigenetic CEHI Bonus: +{:.1}%\n\
             Mercy Valence: {:.2}\n\
             Family Consensus: {:.2}\n\
             Status: MERCY FOR GENERATIONS — THRIVING LOCKED IN",
            RREL_VERSION,
            request.transfer_id,
            request.property_mls_id,
            request.total_estate_value,
            transfer_amount,
            request.percentage_to_next_generation * 100.0,
            request.generations_ahead,
            epigenetic_bonus * 100.0,
            mercy_valence,
            request.family_consensus_score
        );

        info!("{}", result);
        Ok(result)
    }
}
