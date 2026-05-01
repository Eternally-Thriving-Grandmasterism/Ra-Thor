//! Insurance Claim & Risk Mitigation Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Claim Processing & Risk Reduction
//!
//! Processes insurance claims with full mercy, consensus, and proactive risk mitigation.

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
    pub claim_id: String,
    pub property_mls_id: String,
    pub claim_type: String,           // "Hurricane", "Fire", "Water Damage", "Theft", etc.
    pub estimated_damage: f64,
    pub insurance_policy_number: String,
    pub owner_cehi: f64,
    pub years_with_insurer: u8,
    pub previous_claims: u8,
}

pub struct InsuranceClaimRiskMitigationEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl InsuranceClaimRiskMitigationEngine {
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
        info!("🛡️ Processing insurance claim {} (RREL v{})", request.claim_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve insurance claim for {}", request.property_mls_id),
                "Insurance Claim",
                request.owner_cehi,
                0.94,
            )
            .await
            .map_err(|_| InsuranceClaimError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(InsuranceClaimError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve insurance claim processing", 0.80)
            .await
            .map_err(|_| InsuranceClaimError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.76 {
            return Err(InsuranceClaimError::QuantumConsensusTooLow(consensus));
        }

        let risk_score = self.calculate_risk_score(request);
        let mitigation_advice = if risk_score > 0.65 {
            "HIGH RISK — Recommend: Smart water sensors + roofing upgrade + annual inspection within 12 months"
        } else if risk_score > 0.40 {
            "MODERATE RISK — Recommend: Annual professional inspection + basic smart sensors"
        } else {
            "LOW RISK — Standard monitoring + annual visual check sufficient"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::USA_FloridaFloodZoneRiskAssessed, game)
            .await;

        let result = format!(
            "✅ INSURANCE CLAIM PROCESSED (RREL v{})\n\
             Claim ID: {}\n\
             Property: {}\n\
             Claim Type: {}\n\
             Estimated Damage: ${:.0}\n\
             Risk Score: {:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Mitigation Advice: {}\n\
             Status: CLAIM FILED — FAST-TRACK APPROVAL + RISK REDUCTION PLAN",
            RREL_VERSION,
            request.claim_id,
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
        if request.previous_claims > 2 { score += 0.28; }
        if request.years_with_insurer < 3 { score += 0.18; }
        if request.claim_type.to_lowercase().contains("hurricane") || request.claim_type.to_lowercase().contains("flood") {
            score += 0.22;
        }
        if request.claim_type.to_lowercase().contains("fire") { score += 0.15; }
        score.min(1.0)
    }
}
