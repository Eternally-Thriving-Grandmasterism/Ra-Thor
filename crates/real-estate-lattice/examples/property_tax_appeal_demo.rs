//! Property Tax Appeal Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm property tax appeal with estimated savings

use real_estate_lattice::property_tax_appeal::{PropertyTaxAppealEngine, PropertyTaxAppealRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           📉 RREL PROPERTY TAX APPEAL DEMO — v0.5.21                      ║");
    println!("║   Mercy-Gated • Quantum Swarm • Tax Savings Optimization                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut tax_appeal_engine = PropertyTaxAppealEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = PropertyTaxAppealRequest {
        property_mls_id: "TX-2026-0429-7721".to_string(),
        current_assessed_value: 1250000.0,
        proposed_assessed_value: 1080000.0,
        reason: "Comparable sales show 14% over-assessment + recent market decline".to_string(),
        owner_cehi: 8.9,
        years_owned: 7,
    };

    println!("📋 Processing property tax appeal for {}...", request.property_mls_id);

    let result = tax_appeal_engine.process_tax_appeal(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ PROPERTY TAX APPEAL DEMO COMPLETE — STRONG CASE FILED         ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
