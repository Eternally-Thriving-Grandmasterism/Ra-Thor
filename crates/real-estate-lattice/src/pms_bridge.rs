//! PMS Bridge for Ra-Thor Real Estate Lattice (RREL)
//! Bidirectional sync + mercy-gated validation for Yardi, RealPage, AppFolio, Entrata

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, PmsError, WorldImpactType};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PmsProvider {
    Yardi,
    RealPage,
    AppFolio,
    Entrata,
}

#[derive(Debug, Error)]
pub enum RrelError {
    #[error("Mercy valence too low: {0}")]
    MercyRejection(f64),
    #[error("Quantum swarm consensus failed: {0}")]
    SwarmConsensusFailed(String),
    #[error("PMS API error: {0}")]
    PmsApiError(String),
    #[error(transparent)]
    PmsError(#[from] PmsError),
}

pub struct PmsBridge {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl PmsBridge {
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

    pub async fn process_webhook(
        &mut self,
        provider: PmsProvider,
        payload: &str,
        game: &mut patsagi_councils::PowrushGame,
    ) -> Result<String, RrelError> {
        info!("RREL v{} — Processing {} webhook", RREL_VERSION, format!("{:?}", provider));

        // Step 1: Mercy Gate Check
        let valence = self.mercy_engine.evaluate_action(payload).await?;
        if valence < 0.82 {
            warn!("Mercy valence {:.2} < 0.82 — rejecting action", valence);
            return Err(RrelError::MercyRejection(valence));
        }

        // Step 2: Quantum Swarm Consensus
        let consensus = self.quantum_swarm.reach_consensus(payload, 13).await?;
        if consensus < 0.75 {
            return Err(RrelError::SwarmConsensusFailed(format!("Consensus {:.2} < 0.75", consensus)));
        }

        // Step 3: Map to WorldImpactType + Apply
        let impact = match provider {
            PmsProvider::Yardi | PmsProvider::RealPage => WorldImpactType::PMS_TenantApplicationApproved,
            PmsProvider::AppFolio | PmsProvider::Entrata => WorldImpactType::PMS_MaintenanceRequestResolved,
        };

        let result = self.world_governance
            .apply_world_impact(impact, game)
            .await?;

        info!("✅ RREL action approved — Mercy: {:.2} | Swarm: {:.1}%", valence, consensus * 100.0);
        Ok(format!(
            "RREL v{} — {} action approved (Mercy: {:.2}, Swarm: {:.1}%)",
            RREL_VERSION, format!("{:?}", provider), valence, consensus * 100.0
        ))
    }

    // Placeholder for future bidirectional sync methods
    pub async fn sync_yardi(&self) -> Result<(), RrelError> { Ok(()) }
    pub async fn sync_realpage(&self) -> Result<(), RrelError> { Ok(()) }
    pub async fn sync_appfolio(&self) -> Result<(), RrelError> { Ok(()) }
    pub async fn sync_entrata(&self) -> Result<(), RrelError> { Ok(()) }
}
