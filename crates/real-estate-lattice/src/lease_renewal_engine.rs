//! Lease Renewal Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Lease Renewal Optimization
//!
//! Rewards loyal tenants, maximizes occupancy, and strengthens long-term relationships.

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
    pub renewal_id: String,
    pub property_mls_id: String,
    pub tenant_id: String,
    pub current_rent: f64,
    pub years_as_tenant: u8,
    pub tenant_cehi: f64,
    pub payment_history_score: f64, // 0.0 - 10.0
    pub requested_renewal_term_months: u16,
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
        info!("📝 Processing lease renewal {} (RREL v{})", request.renewal_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Approve lease renewal for {}", request.tenant_id),
                "Lease Renewal",
                request.tenant_cehi,
                0.95,
            )
            .await
            .map_err(|_| LeaseRenewalError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(LeaseRenewalError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve lease renewal terms", 0.81)
            .await
            .map_err(|_| LeaseRenewalError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.77 {
            return Err(LeaseRenewalError::QuantumConsensusTooLow(consensus));
        }

        // Calculate loyalty discount
        let loyalty_discount = self.calculate_renewal_discount(request);

        let new_rent = request.current_rent * (1.0 - loyalty_discount);
        let annual_savings = (request.current_rent - new_rent) * 12.0;

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::TenantEvictionPreventedViaMercy, game)
            .await;

        let result = format!(
            "✅ LEASE RENEWAL PROCESSED (RREL v{})\n\
             Renewal ID: {}\n\
             Property: {}\n\
             Tenant: {}\n\
             Current Rent: ${:.0}/mo\n\
             Loyalty Discount: {:.1}%\n\
             New Monthly Rent: ${:.0}\n\
             Annual Tenant Savings: ${:.0}\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Recommended Term: {} months\n\
             Status: MERCY-ALIGNED • LOYALTY REWARDED",
            RREL_VERSION,
            request.renewal_id,
            request.property_mls_id,
            request.tenant_id,
            request.current_rent,
            loyalty_discount * 100.0,
            new_rent,
            annual_savings,
            mercy_valence,
            consensus,
            request.requested_renewal_term_months
        );

        info!("{}", result);
        Ok(result)
    }

    fn calculate_renewal_discount(&self, request: &LeaseRenewalRequest) -> f64 {
        let mut discount = 0.0;
        if request.years_as_tenant >= 5 { discount += 0.08; }
        if request.years_as_tenant >= 3 { discount += 0.04; }
        if request.tenant_cehi >= 8.5 { discount += 0.05; }
        if request.payment_history_score >= 9.0 { discount += 0.04; }
        discount.min(0.18) // Max 18% loyalty discount
    }
}
