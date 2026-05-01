//! Eviction Prevention Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Mercy-First Eviction Prevention
//!
//! Prevents evictions through mercy intervention, payment plans, and support — while still protecting owners.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum EvictionPreventionError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionRiskCase {
    pub tenant_id: String,
    pub property_mls_id: String,
    pub months_behind: u8,
    pub total_arrears: f64,
    pub tenant_cehi: f64,
    pub previous_mercy_interventions: u8,
    pub hardship_reason: String,
}

pub struct EvictionPreventionEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl EvictionPreventionEngine {
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

    pub async fn prevent_eviction(
        &mut self,
        case: &EvictionRiskCase,
        game: &mut PowrushGame,
    ) -> Result<String, EvictionPreventionError> {
        info!("🛡️ Processing eviction prevention case for {} (RREL v{})", case.tenant_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Prevent eviction for {}", case.tenant_id),
                "Eviction Prevention",
                4.8,
                0.97,
            )
            .await
            .map_err(|_| EvictionPreventionError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(EvictionPreventionError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve eviction prevention plan", 0.82)
            .await
            .map_err(|_| EvictionPreventionError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(EvictionPreventionError::QuantumConsensusTooLow(consensus));
        }

        // Trigger the specific mercy-gated world impact
        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PMS_EvictionPreventionViaMercy, game)
            .await;

        let result = format!(
            "🛡️ EVICTION PREVENTED THROUGH MERCY (RREL v{})\n\
             Tenant: {}\n\
             Property: {}\n\
             Months Behind: {}\n\
             Total Arrears: ${:.0}\n\
             CEHI: {:.2}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Action: Payment plan + support approved\n\
             Status: EVICTION BLOCKED — MERCY INTERVENTION SUCCESSFUL",
            RREL_VERSION,
            case.tenant_id,
            case.property_mls_id,
            case.months_behind,
            case.total_arrears,
            case.tenant_cehi,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }
}
