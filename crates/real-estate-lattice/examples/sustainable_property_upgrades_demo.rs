//! Sustainable Property Upgrades Demo — RREL v0.5.21
//! Demonstrates mercy-gated green certification and sustainable upgrade processing

use real_estate_lattice::sustainable_property_upgrades_engine::{SustainablePropertyUpgradesEngine, SustainableUpgradeRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🌱 RREL SUSTAINABLE UPGRADES DEMO — v0.5.21                     ║");
    println!("║   Mercy-Gated • Quantum Swarm • Green Certification Acceleration         ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut upgrades_engine = SustainablePropertyUpgradesEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = SustainableUpgradeRequest {
        upgrade_id: "GREEN-2026-0429-9912".to_string(),
        property_mls_id: "ON-2026-0429-8821".to_string(),
        current_green_certification: Some("ENERGY STAR".to_string()),
        proposed_upgrades: vec![
            "Solar panels (8kW)".to_string(),
            "Heat pump HVAC system".to_string(),
            "Attic + wall insulation upgrade".to_string(),
        ],
        estimated_cost: 48500.0,
        expected_cehi_boost: 2.8,
        owner_cehi: 9.2,
    };

    println!("🌱 Processing sustainable upgrade {}...", request.upgrade_id);

    let result = upgrades_engine.process_sustainable_upgrade(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ SUSTAINABLE UPGRADES DEMO COMPLETE — PLANETARY THRIVING       ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
