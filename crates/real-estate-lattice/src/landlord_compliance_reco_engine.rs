//! Landlord Compliance & RECO Enforcement Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Landlord Oversight
//!
//! Enforces RECO compliance, landlord licensing, trust account rules, and tenant protection with full mercy-first governance.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum LandlordComplianceError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("RECO violation detected — immediate enforcement required")]
    RecoViolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandlordComplianceCheck {
    pub check_id: String,
    pub property_mls_id: String,
    pub landlord_id: String,
    pub has_valid_license: bool,
    pub trust_account_compliant: bool,
    pub tenant_complaints_count: u8,
    pub last_inspection_days_ago: u16,
    pub landlord_cehi: f64,
}

pub struct LandlordComplianceRecoEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl LandlordComplianceRecoEngine {
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

    pub async fn process_landlord_compliance(
        &mut self,
        check: &LandlordComplianceCheck,
        game: &mut PowrushGame,
    ) -> Result<String, LandlordComplianceError> {
        info!("⚖️ Processing landlord compliance check {} (RREL v{})", check.check_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Enforce RECO compliance for {}", check.property_mls_id),
                "Landlord Compliance & RECO",
                check.landlord_cehi,
                0.95,
            )
            .await
            .map_err(|_| LandlordComplianceError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(LandlordComplianceError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve RECO enforcement action", 0.81)
            .await
            .map_err(|_| LandlordComplianceError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.77 {
            return Err(LandlordComplianceError::QuantumConsensusTooLow(consensus));
        }

        let mut violations = Vec::new();
        if !check.has_valid_license { violations.push("No valid RECO license"); }
        if !check.trust_account_compliant { violations.push("Trust account non-compliant"); }
        if check.tenant_complaints_count > 3 { violations.push("Excessive tenant complaints"); }

        if !violations.is_empty() {
            let _ = self.world_governance
                .apply_world_impact(WorldImpactType::RECO_ComplaintPreventedViaMercy, game)
                .await;

            return Ok(format!(
                "🚨 RECO VIOLATION DETECTED (RREL v{})\n\
                 Check ID: {}\n\
                 Property: {}\n\
                 Landlord: {}\n\
                 Violations: {}\n\
                 Mercy Valence: {:.2}\n\
                 Council Consensus: {:.2}\n\
                 Action: IMMEDIATE LICENSE SUSPENSION REVIEW + TENANT RELOCATION SUPPORT + FINE\n\
                 Status: LANDLORD HELD ACCOUNTABLE — MERCY ENFORCED",
                RREL_VERSION,
                check.check_id,
                check.property_mls_id,
                check.landlord_id,
                violations.join(", "),
                mercy_valence,
                consensus
            ));
        }

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::TenantEvictionPreventedViaMercy, game)
            .await;

        let result = format!(
            "✅ LANDLORD COMPLIANCE VERIFIED (RREL v{})\n\
             Check ID: {}\n\
             Property: {}\n\
             Landlord: {}\n\
             License: VALID ✓\n\
             Trust Account: COMPLIANT ✓\n\
             Tenant Complaints: {}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: FULLY COMPLIANT — MERCY-ALIGNED LANDLORD",
            RREL_VERSION,
            check.check_id,
            check.property_mls_id,
            check.landlord_id,
            check.tenant_complaints_count,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
