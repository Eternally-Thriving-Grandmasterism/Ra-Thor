//! Multi-Generational Wealth Transfer & Family Office Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Legacy for Generations
//!
//! Transfers wealth across 3+ generations with CEHI inheritance multipliers, epigenetic blessings,
//! tax optimization integration, and mercy-first governance that maximizes thriving for all.

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
pub struct MultiGenerationalWealthRequest {
    pub transfer_id: String,
    pub property_mls_id: String,
    pub current_owner_cehi: f64,
    pub next_generation_cehi: f64,
    pub generation_3_cehi: Option<f64>,
    pub total_estate_value: f64,
    pub percentage_to_next_generation: f64,
    pub generations_ahead: u8, // 1, 2, or 3
    pub family_consensus_score: f64,
    pub integrate_tax_optimization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGenerationalWealthReport {
    pub recommended_distribution: String,
    pub epigenetic_cehi_bonus: f64,
    pub estimated_tax_savings: f64,
    pub projected_thriving_50_years: f64,
    pub projected_thriving_150_years: f64,
    pub mercy_valence: f64,
    pub council_consensus: f64,
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

    pub async fn process_multi_generational_transfer(
        &mut self,
        request: &MultiGenerationalWealthRequest,
        game: &mut PowrushGame,
    ) -> Result<MultiGenerationalWealthReport, WealthTransferError> {
        info!("🌳 Processing multi-generational wealth transfer {} (RREL v{})", request.transfer_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve multi-generational transfer for {}", request.property_mls_id),
                "Multi-Generational Wealth Transfer",
                (request.current_owner_cehi + request.next_generation_cehi) / 2.0,
                0.96,
            )
            .await
            .map_err(|_| WealthTransferError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(WealthTransferError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve 3-generation wealth transfer plan", 0.82)
            .await
            .map_err(|_| WealthTransferError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(WealthTransferError::QuantumConsensusTooLow(consensus));
        }

        // Epigenetic CEHI Inheritance Multiplier
        let epigenetic_bonus = match request.generations_ahead {
            3 => 0.18,
            2 => 0.12,
            _ => 0.06,
        };

        let tax_savings = if request.integrate_tax_optimization {
            request.total_estate_value * 0.11
        } else {
            0.0
        };

        let projected_50y = request.total_estate_value * (1.0 + epigenetic_bonus) * 2.8;
        let projected_150y = projected_50y * 4.2;

        let distribution = format!(
            "Transfer {:.0}% to Generation 1 + {:.0}% epigenetic blessing to Generation 2 + {:.0}% legacy reserve for Generation 3",
            request.percentage_to_next_generation * 100.0,
            epigenetic_bonus * 100.0,
            100.0 - (request.percentage_to_next_generation * 100.0)
        );

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let report = MultiGenerationalWealthReport {
            recommended_distribution: distribution,
            epigenetic_cehi_bonus: epigenetic_bonus,
            estimated_tax_savings: tax_savings,
            projected_thriving_50_years: projected_50y,
            projected_thriving_150_years: projected_150y,
            mercy_valence,
            council_consensus: consensus,
        };

        info!(
            "✅ MULTI-GENERATIONAL TRANSFER COMPLETE (RREL v{}) — Epigenetic Bonus: +{:.1}% | 150-Year Projection: ${:.0}",
            RREL_VERSION, epigenetic_bonus * 100.0, report.projected_thriving_150_years
        );

        Ok(report)
    }
}
