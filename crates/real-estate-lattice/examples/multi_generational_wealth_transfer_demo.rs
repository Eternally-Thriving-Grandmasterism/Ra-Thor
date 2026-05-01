//! Multi-Generational Wealth Transfer Demo — RREL v0.5.21
//! Demonstrates mercy-gated wealth transfer across generations with CEHI inheritance bonuses

use real_estate_lattice::multi_generational_wealth_transfer_engine::{MultiGenerationalWealthTransferEngine, WealthTransferRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🌳 RREL MULTI-GENERATIONAL WEALTH TRANSFER DEMO — v0.5.21        ║");
    println!("║   Mercy-Gated • Quantum Swarm • Legacy for Generations                   ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut wealth_engine = MultiGenerationalWealthTransferEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = WealthTransferRequest {
        transfer_id: "LEGACY-2026-0429-9912".to_string(),
        property_mls_id: "ON-2026-0429-8821".to_string(),
        current_owner_cehi: 9.3,
        next_generation_cehi: 8.7,
        total_estate_value: 4850000.0,
        percentage_to_next_generation: 0.65,
        generations_ahead: 3,
        family_consensus_score: 0.91,
    };

    println!("🌳 Processing multi-generational wealth transfer {}...", request.transfer_id);

    let result = wealth_engine.process_wealth_transfer(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ MULTI-GENERATIONAL WEALTH TRANSFER DEMO COMPLETE — MERCY FOR  ║");
    println!("║                          GENERATIONS LOCKED IN                            ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
