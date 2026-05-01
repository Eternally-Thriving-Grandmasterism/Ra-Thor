//! Sustainable Property Upgrades & Green Certification Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Green Building Acceleration
//!
//! Helps owners upgrade to sustainable standards with CEHI incentives and mercy-first oversight.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum SustainableUpgradeError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SustainableUpgradeRequest {
    pub upgrade_id: String,
    pub property_mls_id: String,
    pub current_green_certification: Option<String>, // e.g. "ENERGY STAR", "LEED Silver", None
    pub proposed_upgrades: Vec<String>,              // e.g. ["Solar panels", "Heat pump", "Insulation upgrade"]
    pub estimated_cost: f64,
    pub expected_cehi_boost: f64,
    pub owner_cehi: f64,
}

pub struct SustainablePropertyUpgradesEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl SustainablePropertyUpgradesEngine {
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

    pub async fn process_sustainable_upgrade(
        &mut self,
        request: &SustainableUpgradeRequest,
        game: &mut PowrushGame,
    ) -> Result<String, SustainableUpgradeError> {
        info!("🌱 Processing sustainable upgrade {} (RREL v{})", request.upgrade_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve green upgrade for {}", request.property_mls_id),
                "Sustainable Property Upgrades",
                request.owner_cehi,
                0.94,
            )
            .await
            .map_err(|_| SustainableUpgradeError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.88 {
            return Err(SustainableUpgradeError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve sustainable upgrade plan", 0.80)
            .await
            .map_err(|_| SustainableUpgradeError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.76 {
            return Err(SustainableUpgradeError::QuantumConsensusTooLow(consensus));
        }

        let new_certification = if request.proposed_upgrades.iter().any(|u| u.to_lowercase().contains("solar") || u.to_lowercase().contains("heat pump")) {
            "LEED Gold / ENERGY STAR Certified"
        } else {
            "ENERGY STAR Certified"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ SUSTAINABLE UPGRADE APPROVED (RREL v{})\n\
             Upgrade ID: {}\n\
             Property: {}\n\
             Current Certification: {}\n\
             Proposed Upgrades: {}\n\
             Estimated Cost: ${:.0}\n\
             Expected CEHI Boost: +{:.1}\n\
             New Certification: {}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: MERCY-ALIGNED • PLANETARY THRIVING ACCELERATED",
            RREL_VERSION,
            request.upgrade_id,
            request.property_mls_id,
            request.current_green_certification.clone().unwrap_or_else(|| "None".to_string()),
            request.proposed_upgrades.join(", "),
            request.estimated_cost,
            request.expected_cehi_boost,
            new_certification,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
