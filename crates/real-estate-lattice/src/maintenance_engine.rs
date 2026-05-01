//! Maintenance & Repair Request Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Maintenance Prioritization
//!
//! Allows the 13+ PATSAGi Councils to approve and prioritize maintenance requests with full mercy and regulatory alignment.

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
    pub urgency: u8,           // 1 = low, 5 = emergency
    pub tenant_joy_impact: f64, // Estimated joy loss if not fixed
    pub cehi_impact: f64,       // Estimated CEHI impact
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
                4.8,
                0.97,
            )
            .await
            .map_err(|_| MaintenanceError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.88 {
            return Err(MaintenanceError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve maintenance request", 0.82)
            .await
            .map_err(|_| MaintenanceError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(MaintenanceError::QuantumConsensusTooLow(consensus));
        }

        // Calculate priority score
        let priority_score = self.calculate_priority_score(request);

        // Trigger world impact based on urgency
        if request.urgency >= 4 {
            let _ = self.world_governance
                .apply_world_impact(WorldImpactType::PMS_MaintenanceRequestResolved, game)
                .await;
        }

        let result = format!(
            "✅ MAINTENANCE REQUEST APPROVED (RREL v{})\n\
             Request ID: {}\n\
             Property: {}\n\
             Tenant: {}\n\
             Urgency: {}/5\n\
             Priority Score: {:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: APPROVED & SCHEDULED",
            RREL_VERSION,
            request.request_id,
            request.property_mls_id,
            request.tenant_id,
            request.urgency,
            priority_score,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }

    fn calculate_priority_score(&self, request: &MaintenanceRequest) -> f64 {
        let urgency_weight = request.urgency as f64 * 0.35;
        let joy_weight = request.tenant_joy_impact * 0.30;
        let cehi_weight = request.cehi_impact * 0.25;
        let base = 0.10;

        (urgency_weight + joy_weight + cehi_weight + base).min(1.0)
    }
}
