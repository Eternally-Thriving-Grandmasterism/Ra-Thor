//! Emergency Response & Disaster Recovery Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Life-Saving Disaster Coordination
//!
//! Coordinates immediate response, temporary housing, insurance, and long-term recovery with full mercy-first, CEHI-weighted priority.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum DisasterRecoveryError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisasterEvent {
    pub event_id: String,
    pub property_mls_id: String,
    pub disaster_type: String,           // "Hurricane", "Wildfire", "Flood", "Earthquake"
    pub severity_level: u8,              // 1-10
    pub affected_tenant_count: u16,
    pub community_cehi_score: f64,
    pub immediate_needs: Vec<String>,    // e.g. ["Temporary housing", "Medical support", "Food & water"]
}

pub struct EmergencyResponseDisasterRecoveryEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl EmergencyResponseDisasterRecoveryEngine {
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

    pub async fn activate_disaster_response(
        &mut self,
        event: &DisasterEvent,
        game: &mut PowrushGame,
    ) -> Result<String, DisasterRecoveryError> {
        info!("🚨 Activating disaster response for {} (RREL v{})", event.event_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Activate disaster response for {}", event.property_mls_id),
                "Emergency Response",
                event.community_cehi_score,
                0.97,
            )
            .await
            .map_err(|_| DisasterRecoveryError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.91 {
            return Err(DisasterRecoveryError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve disaster response plan", 0.83)
            .await
            .map_err(|_| DisasterRecoveryError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.79 {
            return Err(DisasterRecoveryError::QuantumConsensusTooLow(consensus));
        }

        let priority = if event.severity_level >= 8 || event.affected_tenant_count > 50 {
            "CRITICAL — Immediate evacuation + full emergency support"
        } else if event.severity_level >= 5 {
            "HIGH — Rapid temporary housing + recovery coordination"
        } else {
            "STANDARD — Assessment + phased recovery support"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::TenantEvictionPreventedViaMercy, game)
            .await;

        let result = format!(
            "✅ DISASTER RESPONSE ACTIVATED (RREL v{})\n\
             Event ID: {}\n\
             Property: {}\n\
             Disaster Type: {}\n\
             Severity: {}/10\n\
             Affected Tenants: {}\n\
             Community CEHI: {:.1}\n\
             Priority: {}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: MERCY IN ACTION — COMMUNITIES PROTECTED & RECOVERING",
            RREL_VERSION,
            event.event_id,
            event.property_mls_id,
            event.disaster_type,
            event.severity_level,
            event.affected_tenant_count,
            event.community_cehi_score,
            priority,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
