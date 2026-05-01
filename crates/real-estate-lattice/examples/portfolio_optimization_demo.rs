//! Portfolio Optimization Demo — RREL v0.5.21
//! Demonstrates mercy-gated portfolio optimization with CEHI-weighted recommendations

use real_estate_lattice::portfolio_optimization_engine::{PortfolioOptimizationEngine, PortfolioOptimizationRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           📈 RREL PORTFOLIO OPTIMIZATION DEMO — v0.5.21                   ║");
    println!("║   Mercy-Gated • Quantum Swarm • Intelligent Asset Allocation             ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut portfolio_engine = PortfolioOptimizationEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = PortfolioOptimizationRequest {
        portfolio_id: "ALPHA-2026-0429-001".to_string(),
        total_value: 12450000.0,
        number_of_properties: 14,
        average_cehi: 8.4,
        cash_reserve: 285000.0,
        debt_ratio: 0.48,
        market_trend_score: 0.72,
        risk_tolerance: 0.65,
    };

    println!("📊 Optimizing portfolio {}...", request.portfolio_id);

    let recommendation = portfolio_engine.optimize_portfolio(&request, &mut game).await?;

    println!("\n✅ RECOMMENDATION RECEIVED:");
    println!("   Action: {}", recommendation.recommended_action);
    println!("   Expected Annual Return: {:.1}%", recommendation.expected_annual_return);
    println!("   Risk Reduction: {:.0}%", recommendation.risk_reduction * 100.0);
    println!("   Confidence: {:.1}%", recommendation.confidence * 100.0);
    println!("   Mercy-Aligned: {}", recommendation.mercy_aligned);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ PORTFOLIO OPTIMIZATION DEMO COMPLETE — MERCY VERIFIED         ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
