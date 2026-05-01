//! Predictive Market Valuation & Investment Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Property Valuation & Strategy
//!
//! Predicts future values, rental yields, and optimal investment decisions with full ethical oversight.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum MarketValuationError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketValuationRequest {
    pub valuation_id: String,
    pub property_mls_id: String,
    pub current_value: f64,
    pub location_cehi_score: f64,      // Higher = more ethical/desirable area
    pub rental_yield_current: f64,
    pub market_trend_score: f64,       // -1.0 to +1.0
    pub years_held: u8,
    pub investor_cehi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValuationReport {
    pub predicted_value_12m: f64,
    pub predicted_value_36m: f64,
    pub recommended_action: String,
    pub confidence: f64,
    pub expected_annual_yield: f64,
}

pub struct PredictiveMarketValuationEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl PredictiveMarketValuationEngine {
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

    pub async fn generate_valuation_report(
        &mut self,
        request: &MarketValuationRequest,
        game: &mut PowrushGame,
    ) -> Result<ValuationReport, MarketValuationError> {
        info!("📈 Generating predictive market valuation {} (RREL v{})", request.valuation_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Generate valuation for {}", request.property_mls_id),
                "Market Valuation",
                request.investor_cehi,
                0.93,
            )
            .await
            .map_err(|_| MarketValuationError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.87 {
            return Err(MarketValuationError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve market valuation recommendation", 0.79)
            .await
            .map_err(|_| MarketValuationError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.75 {
            return Err(MarketValuationError::QuantumConsensusTooLow(consensus));
        }

        // Predictive model (simplified but powerful)
        let growth_factor = 1.0 + (request.market_trend_score * 0.08) + (request.location_cehi_score - 7.0) * 0.015;
        let predicted_12m = request.current_value * growth_factor;
        let predicted_36m = predicted_12m * (1.0 + (request.market_trend_score * 0.12));

        let action = if request.market_trend_score > 0.6 && request.location_cehi_score > 8.0 {
            "STRONG BUY — High-growth ethical area"
        } else if request.market_trend_score < -0.4 {
            "HOLD or SELL — Market cooling"
        } else {
            "HOLD — Stable with steady yield"
        };

        let expected_yield = request.rental_yield_current * (1.0 + (request.location_cehi_score - 7.0) * 0.02);

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let report = ValuationReport {
            predicted_value_12m: predicted_12m,
            predicted_value_36m: predicted_36m,
            recommended_action: action.to_string(),
            confidence: (mercy_valence + consensus) / 2.0,
            expected_annual_yield: expected_yield,
        };

        info!(
            "✅ VALUATION COMPLETE (RREL v{}) — Predicted 12m: ${:.0} | Action: {}",
            RREL_VERSION, report.predicted_value_12m, report.recommended_action
        );

        Ok(report)
    }
}
