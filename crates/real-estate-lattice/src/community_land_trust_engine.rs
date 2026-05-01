//! Community Land Trust & Affordable Housing Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Permanent Affordable Housing Protection
//!
//! Creates, governs, and protects Community Land Trusts with mercy-first, multi-generational oversight.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum CommunityLandTrustError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityLandTrustRequest {
    pub trust_id: String,
    pub property_mls_id: String,
    pub community_cehi_score: f64,
    pub number_of_units: u16,
    pub percentage_affordable: f64, // e.g. 0.80 = 80% permanently affordable
    pub land_lease_term_years: u16,
    pub proposed_resale_formula: String, // e.g. "Indexed to AMI + 3%"
}

pub struct CommunityLandTrustEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl CommunityLandTrustEngine {
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

    pub async fn establish_community_land_trust(
        &mut self,
        request: &CommunityLandTrustRequest,
        game: &mut PowrushGame,
    ) -> Result<String, CommunityLandTrustError> {
        info!("🏘️ Establishing Community Land Trust {} (RREL v{})", request.trust_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve CLT for {}", request.property_mls_id),
                "Community Land Trust",
                request.community_cehi_score,
                0.96,
            )
            .await
            .map_err(|_| CommunityLandTrustError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(CommunityLandTrustError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve Community Land Trust establishment", 0.82)
            .await
            .map_err(|_| CommunityLandTrustError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(CommunityLandTrustError::QuantumConsensusTooLow(consensus));
        }

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ COMMUNITY LAND TRUST ESTABLISHED (RREL v{})\n\
             Trust ID: {}\n\
             Property: {}\n\
             Affordable Units: {}/{} ({}%)\n\
             Land Lease Term: {} years\n\
             Resale Formula: {}\n\
             Community CEHI Score: {:.1}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: PERMANENTLY AFFORDABLE — MERCY FOR GENERATIONS",
            RREL_VERSION,
            request.trust_id,
            request.property_mls_id,
            (request.number_of_units as f64 * request.percentage_affordable) as u16,
            request.number_of_units,
            (request.percentage_affordable * 100.0) as u16,
            request.land_lease_term_years,
            request.proposed_resale_formula,
            request.community_cehi_score,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
