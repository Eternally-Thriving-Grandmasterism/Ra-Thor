//! Climate Resilience & Disaster Preparedness Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Climate Protection
//!
//! Helps properties prepare for and recover from climate events with full mercy-first, community-centered oversight.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum ClimateResilienceError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateResilienceRequest {
    pub resilience_id: String,
    pub property_mls_id: String,
    pub climate_risk_score: f64,        // 0.0 - 1.0 (higher = more vulnerable)
    pub community_cehi_score: f64,
    pub current_protections: Vec<String>, // e.g. ["Flood barriers", "Wildfire defensible space"]
    pub proposed_upgrades: Vec<String>,
    pub estimated_cost: f64,
    pub expected_cehi_protection_boost: f64,
}

pub struct ClimateResilienceEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl ClimateResilienceEngine {
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

    pub async fn process_climate_resilience_plan(
        &mut self,
        request: &ClimateResilienceRequest,
        game: &mut PowrushGame,
    ) -> Result<String, ClimateResilienceError> {
        info!("🌍 Processing climate resilience plan {} (RREL v{})", request.resilience_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve climate resilience for {}", request.property_mls_id),
                "Climate Resilience",
                request.community_cehi_score,
                0.95,
            )
            .await
            .map_err(|_| ClimateResilienceError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(ClimateResilienceError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve climate resilience plan", 0.81)
            .await
            .map_err(|_| ClimateResilienceError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.77 {
            return Err(ClimateResilienceError::QuantumConsensusTooLow(consensus));
        }

        let new_protection_level = if request.climate_risk_score > 0.75 {
            "HIGH — Full hardened resilience (flood + fire + heat)"
        } else if request.climate_risk_score > 0.45 {
            "MEDIUM — Targeted upgrades + early warning systems"
        } else {
            "LOW — Maintenance + monitoring"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ CLIMATE RESILIENCE PLAN APPROVED (RREL v{})\n\
             Resilience ID: {}\n\
             Property: {}\n\
             Climate Risk Score: {:.2}\n\
             Community CEHI Score: {:.1}\n\
             Current Protections: {}\n\
             Proposed Upgrades: {}\n\
             Estimated Cost: ${:.0}\n\
             Expected CEHI Protection Boost: +{:.1}\n\
             New Protection Level: {}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: MERCY FOR THE PLANET — COMMUNITIES PROTECTED",
            RREL_VERSION,
            request.resilience_id,
            request.property_mls_id,
            request.climate_risk_score,
            request.community_cehi_score,
            request.current_protections.join(", "),
            request.proposed_upgrades.join(", "),
            request.estimated_cost,
            request.expected_cehi_protection_boost,
            new_protection_level,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
