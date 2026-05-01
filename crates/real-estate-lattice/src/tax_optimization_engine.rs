//! Real Estate Tax Optimization Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Ethical Tax Strategy for Generational Thriving
//!
//! Delivers mercy-first, CEHI-weighted tax optimization with full ethical oversight and long-term thriving focus.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum TaxOptimizationError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxOptimizationRequest {
    pub request_id: String,
    pub property_mls_id: String,
    pub current_year_income: f64,
    pub property_basis: f64,
    pub years_held: u8,
    pub investor_cehi: f64,
    pub has_green_features: bool,
    pub opportunity_zone_eligible: bool,
    pub integrate_with_wealth_transfer: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxOptimizationReport {
    pub recommended_strategy: String,
    pub estimated_tax_savings: f64,
    pub cehi_alignment_score: f64,
    pub confidence: f64,
    pub next_action: String,
    pub multi_gen_impact: Option<String>,
}

pub struct TaxOptimizationEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl TaxOptimizationEngine {
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

    pub async fn generate_tax_optimization_report(
        &mut self,
        request: &TaxOptimizationRequest,
        game: &mut PowrushGame,
    ) -> Result<TaxOptimizationReport, TaxOptimizationError> {
        info!("🧾 Generating tax optimization report {} (RREL v{})", request.request_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Optimize taxes for {}", request.property_mls_id),
                "Tax Optimization",
                request.investor_cehi,
                0.95,
            )
            .await
            .map_err(|_| TaxOptimizationError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.89 {
            return Err(TaxOptimizationError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve tax optimization strategy", 0.81)
            .await
            .map_err(|_| TaxOptimizationError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.77 {
            return Err(TaxOptimizationError::QuantumConsensusTooLow(consensus));
        }

        // Mercy-aligned, CEHI-weighted strategy selection
        let (strategy, savings, cehi_score, action, multi_gen) = if request.opportunity_zone_eligible && request.years_held >= 5 {
            (
                "1031 Exchange into Opportunity Zone + Green Energy Credits + 3-Gen Legacy Planning",
                request.current_year_income * 0.19,
                9.5,
                "Initiate 1031 exchange within 180 days + apply for federal green tax credits + structure for multi-generational transfer",
                Some("Epigenetic CEHI bonus of +18% applied across 3 generations".to_string())
            )
        } else if request.has_green_features {
            (
                "Accelerated Depreciation + Federal Green Energy Tax Credits + CEHI Tenant Bonus",
                request.current_year_income * 0.13,
                9.2,
                "File Form 5695 + claim 30% solar + heat pump credits this tax year + pass 15% of savings to tenants via rent stabilization",
                None
            )
        } else if request.years_held >= 7 {
            (
                "Cost Segregation Study + Bonus Depreciation + Multi-Gen Wealth Structuring",
                request.current_year_income * 0.10,
                8.6,
                "Commission cost segregation study before year-end + align with 3-generation wealth transfer plan",
                Some("Tax savings structured to support 150-year family thriving projection".to_string())
            )
        } else {
            (
                "Standard Depreciation + Maximize Deductions + Ethical Tenant Support",
                request.current_year_income * 0.07,
                8.1,
                "Review all eligible deductions with tax professional + consider small CEHI-based tenant relief fund",
                None
            )
        };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let report = TaxOptimizationReport {
            recommended_strategy: strategy.to_string(),
            estimated_tax_savings: savings,
            cehi_alignment_score: cehi_score,
            confidence: (mercy_valence + consensus) / 2.0,
            next_action: action.to_string(),
            multi_gen_impact: multi_gen,
        };

        info!(
            "✅ TAX OPTIMIZATION COMPLETE (RREL v{}) — Strategy: {} | Estimated Savings: ${:.0}",
            RREL_VERSION, report.recommended_strategy, report.estimated_tax_savings
        );

        Ok(report)
    }
}
