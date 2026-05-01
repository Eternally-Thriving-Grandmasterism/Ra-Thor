//! Portfolio Optimization Engine — RREL v0.5.21
//! Mercy-Gated • Quantum Swarm • Intelligent Portfolio Decision Making
//!
//! Recommends optimal real estate portfolio actions with full ethical alignment.

use crate::RREL_VERSION;
use patsagi_councils::{WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum PortfolioOptimizationError {
    #[error("Mercy valence too low: {0:.2}")]
    MercyGateFailed(f64),
    #[error("Quantum swarm consensus too low: {0:.2}")]
    QuantumConsensusTooLow(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioOptimizationRequest {
    pub portfolio_id: String,
    pub total_value: f64,
    pub number_of_properties: u8,
    pub average_cehi: f64,
    pub cash_reserve: f64,
    pub debt_ratio: f64,
    pub market_trend_score: f64, // -1.0 to +1.0
    pub risk_tolerance: f64,     // NEW: 0.0 - 1.0 (merged for modernity)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommended_action: String, // Unified naming
    pub confidence: f64,
    pub expected_annual_return: f64,
    pub risk_reduction: f64,
    pub mercy_aligned: bool,
}

pub struct PortfolioOptimizationEngine {
    mercy_engine: MercyEngine,
    quantum_swarm: QuantumSwarmOrchestrator,
    world_governance: WorldGovernanceEngine,
}

impl PortfolioOptimizationEngine {
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

    pub async fn optimize_portfolio(
        &mut self,
        request: &PortfolioOptimizationRequest,
        game: &mut PowrushGame,
    ) -> Result<OptimizationRecommendation, PortfolioOptimizationError> {
        info!("📈 Optimizing portfolio {} (RREL v{})", request.portfolio_id, RREL_VERSION);

        let mercy_valence = self.mercy_engine
            .evaluate_action(
                &format!("Optimize portfolio {}", request.portfolio_id),
                "Portfolio Optimization",
                request.average_cehi,
                0.96,
            )
            .await
            .map_err(|_| PortfolioOptimizationError::MercyGateFailed(0.0))?;

        if mercy_valence < 0.88 {
            return Err(PortfolioOptimizationError::MercyGateFailed(mercy_valence));
        }

        let consensus = self.quantum_swarm
            .reach_consensus("Approve portfolio optimization recommendation", 0.80)
            .await
            .map_err(|_| PortfolioOptimizationError::QuantumConsensusTooLow(0.0))?;

        if consensus < 0.75 {
            return Err(PortfolioOptimizationError::QuantumConsensusTooLow(consensus));
        }

        // Core optimization logic (preserved from your attached version + risk_tolerance enhancement)
        let action = if request.debt_ratio > 0.65 && request.cash_reserve < 50000.0 {
            "Refinance high-interest debt + build cash reserve"
        } else if request.market_trend_score > 0.6 && request.cash_reserve > 200000.0 && request.risk_tolerance > 0.6 {
            "Acquire 1-2 additional properties in growth markets"
        } else if request.average_cehi < 7.5 {
            "Focus on tenant retention & property upgrades for CEHI boost"
        } else {
            "Hold current portfolio + explore fractional ownership opportunities"
        };

        let expected_return = match action {
            a if a.contains("Acquire") => 9.2,
            a if a.contains("Refinance") => 7.8,
            a if a.contains("Focus") => 6.5,
            _ => 5.9,
        };

        let risk_reduction = if mercy_valence > 0.92 { 0.28 } else { 0.15 };

        let _ = self.world_governance
            .apply_world_impact(WorldImpactType::PortfolioAcquisitionConsensus, game)
            .await;

        let recommendation = OptimizationRecommendation {
            recommended_action: action.to_string(),
            confidence: (mercy_valence + consensus) / 2.0,
            expected_annual_return: expected_return,
            risk_reduction,
            mercy_aligned: true,
        };

        let result = format!(
            "✅ PORTFOLIO OPTIMIZATION COMPLETE (RREL v{})\n\
             Portfolio: {}\n\
             Total Value: ${:.0}\n\
             Recommended Action: {}\n\
             Expected Annual Return: {:.1}%\n\
             Risk Reduction: {:.0}%\n\
             Mercy Valence: {:.2}\n\
             Council Consensus: {:.2}\n\
             Status: MERCY-ALIGNED • QUANTUM-VERIFIED",
            RREL_VERSION,
            request.portfolio_id,
            request.total_value,
            recommendation.recommended_action,
            recommendation.expected_annual_return,
            recommendation.risk_reduction * 100.0,
            mercy_valence,
            consensus
        );

        info!("{}", result);
        Ok(recommendation)
    }
}
