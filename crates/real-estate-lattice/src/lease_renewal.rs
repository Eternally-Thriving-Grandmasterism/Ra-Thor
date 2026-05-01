//! Lease Renewal Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Lease Renewal System
//!
//! Automatically processes lease renewals with mercy-first logic and smart incentives for high-CEHI tenants.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum LeaseRenewalError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaseRenewalRequest {
    pub tenant_id: String,
    pub property_mls_id: String,
    pub current_rent: f64,
    pub lease_end_date: chrono::DateTime<chrono::Utc>,
    pub tenant_cehi: f64,
    pub years_as_tenant: u32,
    pub payment_history_score: f64, // 0.0 – 1.0
}

pub struct LeaseRenewalEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl LeaseRenewalEngine {
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

    pub async fn process_lease_renewal(
        &mut self,
        request: &LeaseRenewalRequest,
        game: &mut PowrushGame,
    ) -> Result<String, LeaseRenewalError> {
        info!("📜 Processing lease renewal for {} (RREL v{})", request.tenant_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Renew lease for {}", request.tenant_id),
                "Lease Renewal",
                4.8,
                0.97,
            )
            .await
            .map_err(|_| LeaseRenewalError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.90 {
            return Err(LeaseRenewalError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve lease renewal", 0.82)
            .await
            .map_err(|_| LeaseRenewalError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.78 {
            return Err(LeaseRenewalError::QuantumConsensusTooLow(consensus));
        }

        // Calculate renewal discount based on loyalty + CEHI
        let discount = self.calculate_renewal_discount(request);
        let new_rent = request.current_rent * (1.0 - discount);

        // Trigger positive world impact
        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PMS_LeaseRenewalWithMercy, game)
            .await;

        let result = format!(
            "✅ LEASE RENEWAL APPROVED (RREL v{})\n\
             Tenant: {}\n\
             Property: {}\n\
             Current Rent: ${:.0}\n\
             New Rent: ${:.0} ({}% discount)\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: RENEWED FOR 12 MONTHS",
            RREL_VERSION,
            request.tenant_id,
            request.property_mls_id,
            request.current_rent,
            new_rent,
            (discount * 100.0) as u32,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(result)
    }

    fn calculate_renewal_discount(&self, request: &LeaseRenewalRequest) -> f64 {
        let loyalty_bonus = (request.years_as_tenant as f64 * 0.015).min(0.12);
        let cehi_bonus = (request.tenant_cehi / 10.0) * 0.08;
        let payment_bonus = request.payment_history_score * 0.05;

        (loyalty_bonus + cehi_bonus + payment_bonus).min(0.20)
    }
}
