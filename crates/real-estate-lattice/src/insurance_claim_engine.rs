//! Insurance Claim & Risk Mitigation Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Insurance Claim Processing
//!
//! Helps owners file, track, and optimize insurance claims with full mercy and regulatory alignment.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum InsuranceClaimError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsuranceClaimRequest {
    pub property_mls_id: String,
    pub claim_type: String,           // e.g. "Hurricane", "Fire", "Water Damage", "Theft"
    pub estimated_damage: f64,
    pub insurance_policy_number: String,
    pub owner_cehi: f64,
    pub years_with_insurer: u32,
    pub previous_claims: u8,
}

pub struct InsuranceClaimEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl InsuranceClaimEngine {
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

    pub async fn process_insurance_claim(
        &mut self,
        request: &InsuranceClaimRequest,
        game: &mut PowrushGame,
    ) -> Result<String, InsuranceClaimError> {
        info!("🛡️ Processing insurance claim for {} (RREL v{})", request.property_mls_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve insurance claim for {}", request.property_mls_id),
                "Insurance Claim",
                4.8,
                0.97,
            )
            .await
            .map_err(|_| InsuranceClaimError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(InsuranceClaimError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve insurance claim processing", 0.82)
            .await
            .map_err(|_| InsuranceClaimError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(InsuranceClaimError::QuantumConsensusTooLow(consensus));
        }

        // Risk mitigation recommendation
        let risk_score = self.calculate_risk_score(request);
        let mitigation_advice = if risk_score > 0.6 {
            "Recommend: Install smart water sensors + upgrade roofing within 18 months"
        } else {
            "Low future risk — standard monitoring recommended"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::USA_FloridaFloodZoneRiskAssessed, game)
            .await;

        let result = format!(
            "✅ INSURANCE CLAIM APPROVED (RREL v{})\n\
             Property: {}\n\
             Claim Type: {}\n\
             Estimated Damage: ${:.0}\n\
             Risk Score: {:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Mitigation Advice: {}\n\
             Status: CLAIM FILED — FAST-TRACK APPROVAL",
            RREL_VERSION,
            request.property_mls_id,
            request.claim_type,
            request.estimated_damage,
            risk_score,
            mercy_valence,
            consensus,
            mitigation_advice
        );

        info!("{}", result);
        Ok(result)
    }

    fn calculate_risk_score(&self, request: &InsuranceClaimRequest) -> f64 {
        let mut score = 0.0;
        if request.previous_claims > 2 { score += 0.25; }
        if request.years_with_insurer < 3 { score += 0.15; }
        if request.claim_type.to_lowercase().contains("hurricane") || request.claim_type.to_lowercase().contains("flood") {
            score += 0.20;
        }
        score.min(1.0)
    }
}
