//! Rental Income Distribution Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm automated rental income distribution to fractional owners

use real_estate_lattice::rental_income_distribution::{RentalIncomeDistributionEngine, RentalDistributionRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           💰 RREL RENTAL DISTRIBUTION DEMO — v0.5.21                      ║");
    println!("║   Mercy-Gated • Quantum Swarm • Fair Income Distribution                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut distribution_engine = RentalIncomeDistributionEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = RentalDistributionRequest {
        property_mls_id: "CA-2026-0429-8847".to_string(),
        total_rent_collected: 12500.0,
        number_of_shares: 10000,
        distribution_date: chrono::Utc::now(),
    };

    println!("📋 Processing rental income distribution for {}...", request.property_mls_id);

    let result = distribution_engine.distribute_rental_income(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ RENTAL DISTRIBUTION DEMO COMPLETE — INCOME DISTRIBUTED        ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
