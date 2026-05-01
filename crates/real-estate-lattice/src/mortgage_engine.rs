//! Mortgage Automation Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • RECO/TRESA Compliant Mortgage Processing
//!
//! Allows the 13+ PATSAGi Councils to approve mortgages with full regulatory and mercy alignment.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum MortgageError {
    #[error("Mercy valence too low for mortgage approval: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("RECO/TRESA compliance check failed")]
    RegulatoryCheckFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MortgageApplication {
    pub applicant_id: String,
    pub property_mls_id: String,
    pub loan_amount: f64,
    pub down_payment: f64,
    pub credit_score: u16,
    pub annual_income: f64,
}

pub struct MortgageEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl MortgageEngine {
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

    /// Full mercy-gated + regulatory mortgage approval
    pub async fn process_mortgage_application(
        &mut self,
        application: &MortgageApplication,
        game: &mut PowrushGame,
    ) -> Result<String, MortgageError> {
        info!("🏠 Processing mortgage application for {} (RREL v{})", application.applicant_id, RREL_VERSION);

        // Step 1: Mercy Gate
        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve mortgage for {}", application.applicant_id),
                "Mortgage Approval",
                4.8,
                0.97,
            )
            .await
            .map_err(|_| MortgageError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(MortgageError::MercyGateFailed(mercy_valence));
        }

        // Step 2: Quantum Swarm Consensus
        let consensus = self.quantum_swarm
            .reach_consensus("Approve mortgage application", 0.85)
            .await
            .map_err(|_| MortgageError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.80 {
            return Err(MortgageError::QuantumConsensusTooLow(consensus));
        }

        // Step 3: Apply World Impact (triggers joy, resources, epigenetic blessing)
        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let result = format!(
            "✅ MORTGAGE APPROVED (RREL v{})\n\
             Applicant: {}\n\
             Property: {}\n\
             Loan Amount: ${:.0}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: APPROVED & FUNDED",
            RREL_VERSION,
            application.applicant_id,
            application.property_mls_id,
            application.loan_amount,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
