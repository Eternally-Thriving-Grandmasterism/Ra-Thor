//! Maintenance & Repair Request Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Maintenance Prioritization
//!
//! Allows the 13+ PATSAGi Councils to approve and prioritize maintenance requests
//! with full mercy, quantum consensus, and predictive optimization.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum MaintenanceError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceRequest {
    pub request_id: String,
    pub property_mls_id: String,
    pub tenant_id: String,
    pub issue_description: String,
    pub urgency: u8,                    // 1 = low, 5 = emergency
    pub tenant_joy_impact: f64,         // Estimated joy loss if not fixed
    pub cehi_impact: f64,               // Estimated CEHI impact
    pub estimated_cost: f64,            // NEW: from predictive version
    pub days_since_last_inspection: u16, // NEW: predictive factor
}

pub struct MaintenanceEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl MaintenanceEngine {
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

    pub async fn process_maintenance_request(
        &mut self,
        request: &MaintenanceRequest,
        game: &mut PowrushGame,
    ) -> Result<String, MaintenanceError> {
        info!("🔧 Processing maintenance request {} (RREL v{})", request.request_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve maintenance for {}", request.tenant_id),
                "Maintenance Approval",
                request.cehi_impact,
                0.95,
            )
            .await
            .map_err(|_| MaintenanceError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(MaintenanceError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve maintenance request", 0.81)
            .await
            .map_err(|_| MaintenanceError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.77 {
            return Err(MaintenanceError::QuantumConsensusTooLow(consensus));
        }

        // Enhanced predictive priority scoring (merged best of both)
        let priority_score = self.calculate_priority_score(request);

        // Trigger appropriate world impact
        if request.urgency >= 4 {
            let _ = self.world_governance
                .apply_world_impact(WorldImpactType::PMS_MaintenanceRequestResolved, game)
                .await;
        } else {
            let _ = self.world_governance
                .apply_world_impact(WorldImpactType::TenantEvictionPreventedViaMercy, game)
                .await;
        }

        let recommended_action = if priority_score > 0.88 {
            "IMMEDIATE — Schedule within 24 hours"
        } else if priority_score > 0.70 {
            "HIGH PRIORITY — Schedule within 72 hours"
        } else {
            "STANDARD — Schedule within 14 days + preventive monitoring"
        };

        let result = format!(
            "✅ MAINTENANCE REQUEST APPROVED (RREL v{})\n\
             Request ID: {}\n\
             Property: {}\n\
             Tenant: {}\n\
             Issue: {}\n\
             Urgency: {}/5\n\
             Priority Score: {:.2}\n\
             Estimated Cost: ${:.0}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Recommended Action: {}\n\
             Expected Tenant Joy Recovery: +{:.1} CEHI points\n\
             Status: APPROVED & SCHEDULED — MERCY VERIFIED",
            RREL_VERSION,
            request.request_id,
            request.property_mls_id,
            request.tenant_id,
            request.issue_description,
            request.urgency,
            priority_score,
            request.estimated_cost,
            mercy_valence,
            consensus,
            recommended_action,
            request.tenant_joy_impact * 0.85
        );

        info!("{}", result);
        Ok(result)
    }

    fn calculate_priority_score(&self, request: &MaintenanceRequest) -> f64 {
        let urgency_weight = (request.urgency as f64) * 0.32;
        let joy_weight = request.tenant_joy_impact * 0.28;
        let cehi_weight = request.cehi_impact * 0.22;
        let inspection_penalty = if request.days_since_last_inspection > 180 { 0.12 } else { 0.0 };
        let base = 0.08;

        (urgency_weight + joy_weight + cehi_weight + inspection_penalty + base).min(1.0)
    }
}
