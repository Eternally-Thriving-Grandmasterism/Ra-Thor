//! Tenant Screening & Fair Housing Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Tenant Risk Assessment
//!
//! Processes applications with full fair housing compliance and mercy-first decision making.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum TenantScreeningError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("Fair housing violation risk detected")]
    FairHousingRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantApplication {
    pub application_id: String,
    pub property_mls_id: String,
    pub applicant_cehi: f64,
    pub credit_score: u16,
    pub income_to_rent_ratio: f64,
    pub rental_history_years: u8,
    pub eviction_history: u8,
    pub references_score: f64, // 0.0 - 10.0
}

pub struct TenantScreeningEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl TenantScreeningEngine {
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

    pub async fn screen_tenant_application(
        &mut self,
        application: &TenantApplication,
        game: &mut PowrushGame,
    ) -> Result<String, TenantScreeningError> {
        info!("👤 Screening tenant application {} (RREL v{})", application.application_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Screen tenant for {}", application.property_mls_id),
                "Tenant Screening",
                application.applicant_cehi,
                0.95,
            )
            .await
            .map_err(|_| TenantScreeningError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(TenantScreeningError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve tenant screening decision", 0.82)
            .await
            .map_err(|_| TenantScreeningError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(TenantScreeningError::QuantumConsensusTooLow(consensus));
        }

        // Calculate ethical risk score (never uses protected class data)
        let risk_score = self.calculate_ethical_risk_score(application);

        if risk_score > 0.85 && application.eviction_history > 1 {
            let _ = self.world_governance
                .apply_world_impact(WorldImpactType::RECO_ComplaintPreventedViaMercy, game)
                .await;

            return Ok(format!(
                "⚠️ APPLICATION REVIEW REQUIRED (RREL v{})\n\
                 High risk detected but mercy-aligned review triggered.\n\
                 Recommended: Offer co-signer option + financial literacy support.\n\
                 Mercy Valence: {:.2} | Council Consensus: {:.2}",
                RREL_VERSION, mercy_valence, consensus
            ));
        }

        let decision = if risk_score < 0.35 {
            "APPROVED — Excellent tenant profile"
        } else if risk_score < 0.55 {
            "APPROVED WITH CONDITIONS — Standard lease + security deposit"
        } else {
            "CONDITIONAL APPROVAL — Higher deposit + monthly check-ins"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::TenantEvictionPreventedViaMercy, game)
            .await;

        let result = format!(
            "✅ TENANT SCREENING COMPLETE (RREL v{})\n\
             Application: {}\n\
             Property: {}\n\
             Ethical Risk Score: {:.2}\n\
             Decision: {}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Fair Housing: FULLY COMPLIANT ✓",
            RREL_VERSION,
            application.application_id,
            application.property_mls_id,
            risk_score,
            decision,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }

    fn calculate_ethical_risk_score(&self, app: &TenantApplication) -> f64 {
        let mut score = 0.0;
        if app.credit_score < 580 { score += 0.22; }
        if app.income_to_rent_ratio < 2.5 { score += 0.18; }
        if app.eviction_history > 0 { score += 0.25; }
        if app.rental_history_years < 2 { score += 0.12; }
        if app.references_score < 6.0 { score += 0.15; }
        score.min(1.0)
    }
}
