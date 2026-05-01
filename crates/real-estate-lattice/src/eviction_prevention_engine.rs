//! Eviction Prevention Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Eviction Prevention
//!
//! Proactively prevents evictions through mercy intervention, payment plans, and hardship support.

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
    pub case_id: String,
    pub property_mls_id: String,
    pub tenant_id: String,
    pub months_behind: u8,
    pub arrears_amount: f64,
    pub tenant_cehi: f64,
    pub hardship_reason: String,
    pub previous_interventions: u8,
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
        info!("🛡️ Processing eviction prevention case {} (RREL v{})", case.case_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Prevent eviction for {}", case.tenant_id),
                "Eviction Prevention",
                case.tenant_cehi,
                0.96,
            )
            .await
            .map_err(|_| EvictionPreventionError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(EvictionPreventionError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve eviction prevention intervention", 0.82)
            .await
            .map_err(|_| EvictionPreventionError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(EvictionPreventionError::QuantumConsensusTooLow(consensus));
        }

        // Generate mercy-aligned intervention plan
        let intervention = if case.months_behind >= 4 {
            "Full 12-month payment plan + 3-month rent reduction + financial counseling referral"
        } else if case.months_behind >= 2 {
            "6-month payment plan + 1-month rent credit + hardship grant application support"
        } else {
            "3-month payment plan + immediate landlord-tenant mediation"
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::TenantEvictionPreventedViaMercy, game)
            .await;

        let result = format!(
            "✅ EVICTION PREVENTION SUCCESSFUL (RREL v{})\n\
             Case ID: {}\n\
             Property: {}\n\
             Tenant: {}\n\
             Months Behind: {}\n\
             Arrears: ${:.0}\n\
             Intervention Plan: {}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Expected CEHI Recovery: +{:.1} points\n\
             Status: EVICTION PREVENTED — MERCY FIRST",
            RREL_VERSION,
            case.case_id,
            case.property_mls_id,
            case.tenant_id,
            case.months_behind,
            case.arrears_amount,
            intervention,
            mercy_valence,
            consensus,
            case.tenant_cehi * 0.6
        );

        info!("{}", result);
        Ok(result)
    }
}
