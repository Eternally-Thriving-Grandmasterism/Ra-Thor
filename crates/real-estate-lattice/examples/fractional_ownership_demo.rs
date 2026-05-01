//! Fractional Ownership Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm tokenized fractional real estate ownership

use real_estate_lattice::fractional_ownership::{FractionalOwnershipEngine, FractionalOwnershipRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🏠 RREL FRACTIONAL OWNERSHIP DEMO — v0.5.21                     ║");
    println!("║   Mercy-Gated • Quantum Swarm • Tokenized Property Shares                ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut fractional_engine = FractionalOwnershipEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = FractionalOwnershipRequest {
        property_mls_id: "CA-2026-0429-8847".to_string(),
        total_property_value: 1250000.0,
        shares_offered: 10000,
        price_per_share: 125.0,
        minimum_investment: 500.0,
    };

    println!("📋 Processing fractional ownership offering for {}...", request.property_mls_id);

    let result = fractional_engine.approve_fractional_offering(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ FRACTIONAL OWNERSHIP DEMO COMPLETE — LIVE ON POWRUSH-MMO      ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
