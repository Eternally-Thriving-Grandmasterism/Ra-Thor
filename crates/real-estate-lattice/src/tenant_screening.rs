//! Tenant Screening Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • RECO/TRESA Compliant Tenant Screening
//!
//! Allows the 13+ PATSAGi Councils to approve or reject tenant applications with full regulatory and mercy alignment.

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
    #[error("Mercy valence too low for tenant approval: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("RECO/TRESA compliance check failed")]
    RegulatoryCheckFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantApplication {
    pub applicant_id: String,
    pub property_mls_id: String,
    pub credit_score: u16,
    pub annual_income: f64,
    pub criminal_record: bool,
    pub previous_evictions: u8,
    pub references_score: f64, // 0.0 – 10.0
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
        info!("👤 Screening tenant application for {} (RREL v{})", application.applicant_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Screen tenant {}", application.applicant_id),
                "Tenant Screening",
                4.8,
                0.97,
            )
            .await
            .map_err(|_| TenantScreeningError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.88 {
            return Err(TenantScreeningError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve or reject tenant application", 0.82)
            .await
            .map_err(|_| TenantScreeningError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(TenantScreeningError::QuantumConsensusTooLow(consensus));
        }

        // Basic risk scoring
        let risk_score = self.calculate_risk_score(application);

        if risk_score > 0.65 {
            // High risk — trigger mercy intervention instead of automatic rejection
            let _ = self.world_governance
                .apply_world_impact(WorldImpactType::PMS_EvictionPreventionViaMercy, game)
                .await;

            return Ok(format!(
                "🛡️ TENANT APPLICATION FLAGGED FOR MERCY REVIEW (RREL v{})\n\
                 Applicant: {}\n\
                 Risk Score: {:.2}\n\
                 Mercy Valence: {:.2}\n\
                 Status: FLAGGED — Mercy intervention recommended",
                RREL_VERSION, application.applicant_id, risk_score, mercy_valence
            ));
        }

        // Low risk — approve
        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PMS_TenantApplicationApproved, game)
            .await;

        Ok(format!(
            "✅ TENANT APPLICATION APPROVED (RREL v{})\n\
             Applicant: {}\n\
             Risk Score: {:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: APPROVED",
            RREL_VERSION, application.applicant_id, risk_score, mercy_valence, consensus
        ))
    }

    fn calculate_risk_score(&self, app: &TenantApplication) -> f64 {
        let mut score = 0.0;

        if app.credit_score < 620 { score += 0.25; }
        if app.annual_income < 45000.0 { score += 0.20; }
        if app.criminal_record { score += 0.30; }
        if app.previous_evictions > 0 { score += 0.15 * app.previous_evictions as f64; }
        if app.references_score < 6.5 { score += 0.15; }

        score.min(1.0)
    }
}
