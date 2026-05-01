//! Tenant Rights & Habitability Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Tenant Protection
//!
//! Enforces habitability standards, tenant rights, and prevents slumlord behavior with full mercy-first oversight.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum TenantRightsError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
    #[error("Habitability violation detected — immediate action required")]
    HabitabilityViolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HabitabilityInspection {
    pub inspection_id: String,
    pub property_mls_id: String,
    pub tenant_id: String,
    pub issues_found: Vec<String>, // e.g. "No heat", "Mold in bathroom", "Broken smoke detector"
    pub tenant_cehi: f64,
    pub days_since_last_inspection: u16,
    pub landlord_response_time_days: u8,
}

pub struct TenantRightsHabitabilityEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl TenantRightsHabitabilityEngine {
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

    pub async fn process_habitability_inspection(
        &mut self,
        inspection: &HabitabilityInspection,
        game: &mut PowrushGame,
    ) -> Result<String, TenantRightsError> {
        info!("🏠 Processing habitability inspection {} (RREL v{})", inspection.inspection_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Enforce tenant rights for {}", inspection.property_mls_id),
                "Tenant Rights & Habitability",
                inspection.tenant_cehi,
                0.96,
            )
            .await
            .map_err(|_| TenantRightsError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(TenantRightsError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve habitability enforcement action", 0.82)
            .await
            .map_err(|_| TenantRightsError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(TenantRightsError::QuantumConsensusTooLow(consensus));
        }

        let severity = inspection.issues_found.len() as f64 * 0.15;
        let urgency_score = (severity + (inspection.days_since_last_inspection as f64 / 365.0)).min(1.0);

        if urgency_score > 0.75 {
            let _ = self.world_governance
                .apply_world_impact(WorldImpactType::RECO_ComplaintPreventedViaMercy, game)
                .await;

            return Ok(format!(
                "🚨 URGENT HABITABILITY VIOLATION (RREL v{})\n\
                 Inspection ID: {}\n\
                 Property: {}\n\
                 Issues: {}\n\
                 Urgency Score: {:.2}\n\
                 Mercy Valence: {:.2}\n\
                 Council Consensus: {:.2}\n\
                 Action: IMMEDIATE REPAIR ORDER + TENANT RELOCATION SUPPORT + LANDLORD PENALTY\n\
                 Status: TENANT RIGHTS PROTECTED — MERCY ENFORCED",
                RREL_VERSION,
                inspection.inspection_id,
                inspection.property_mls_id,
                inspection.issues_found.join(", "),
                urgency_score,
                mercy_valence,
                consensus
            ));
        }

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::TenantEvictionPreventedViaMercy, game)
            .await;

        let result = format!(
            "✅ HABITABILITY INSPECTION PROCESSED (RREL v{})\n\
             Inspection ID: {}\n\
             Property: {}\n\
             Issues Found: {}\n\
             Urgency Score: {:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Recommended Action: Schedule repairs within {} days + tenant compensation\n\
             Status: TENANT RIGHTS PROTECTED — MERCY FIRST",
            RREL_VERSION,
            inspection.inspection_id,
            inspection.property_mls_id,
            inspection.issues_found.join(", "),
            urgency_score,
            mercy_valence,
            consensus,
            if urgency_score > 0.5 { "7" } else { "14" }
        );

        info!("{}", result);
        Ok(result)
    }
}
