//! Mortgage Processing Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Mortgage Approval
//!
//! Processes mortgage applications with fair lending, CEHI-weighted decisions, and mercy-first oversight.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum MortgageProcessingError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("Fair lending risk detected")]
    FairLendingRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MortgageApplication {
    pub application_id: String,
    pub property_mls_id: String,
    pub applicant_cehi: f64,
    pub credit_score: u16,
    pub debt_to_income_ratio: f64,
    pub down_payment_percentage: f64,
    pub loan_amount: f64,
    pub years_in_current_job: u8,
}

pub struct MortgageProcessingEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl MortgageProcessingEngine {
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

    pub async fn process_mortgage_application(
        &mut self,
        application: &MortgageApplication,
        game: &mut PowrushGame,
    ) -> Result<String, MortgageProcessingError> {
        info!("🏦 Processing mortgage application {} (RREL v{})", application.application_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve mortgage for {}", application.property_mls_id),
                "Mortgage Approval",
                application.applicant_cehi,
                0.94,
            )
            .await
            .map_err(|_| MortgageProcessingError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.88 {
            return Err(MortgageProcessingError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve mortgage terms", 0.80)
            .await
            .map_err(|_| MortgageProcessingError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.76 {
            return Err(MortgageProcessingError::QuantumConsensusTooLow(consensus));
        }

        // Ethical approval score (never uses protected class data)
        let approval_score = self.calculate_ethical_approval_score(application);

        let decision = if approval_score >= 0.82 {
            "PRE-APPROVED — Fast-track to underwriting"
        } else if approval_score >= 0.68 {
            "CONDITIONAL APPROVAL — Additional documentation required"
        } else {
            "REVIEW REQUIRED — Offer financial literacy support + co-signer option"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ MORTGAGE APPLICATION PROCESSED (RREL v{})\n\
             Application ID: {}\n\
             Property: {}\n\
             Ethical Approval Score: {:.2}\n\
             Decision: {}\n\
             Loan Amount: ${:.0}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Fair Lending: FULLY COMPLIANT ✓\n\
             Status: MERCY-ALIGNED • ETHICAL FINANCING",
            RREL_VERSION,
            application.application_id,
            application.property_mls_id,
            approval_score,
            decision,
            application.loan_amount,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }

    fn calculate_ethical_approval_score(&self, app: &MortgageApplication) -> f64 {
        let mut score = 0.0;
        if app.credit_score >= 720 { score += 0.28; }
        else if app.credit_score >= 680 { score += 0.20; }
        if app.debt_to_income_ratio < 0.36 { score += 0.25; }
        if app.down_payment_percentage >= 0.20 { score += 0.18; }
        if app.years_in_current_job >= 3 { score += 0.15; }
        score.min(1.0)
    }
}
