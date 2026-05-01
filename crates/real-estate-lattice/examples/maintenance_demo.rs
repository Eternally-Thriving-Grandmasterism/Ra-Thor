//! Maintenance Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm intelligent maintenance request prioritization

use real_estate_lattice::maintenance_engine::{MaintenanceEngine, MaintenanceRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🔧 RREL MAINTENANCE DEMO — v0.5.21                              ║");
    println!("║   Mercy-Gated • Quantum Swarm • Intelligent Prioritization               ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut maintenance_engine = MaintenanceEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = MaintenanceRequest {
        request_id: "MAINT-2026-0429-8847".to_string(),
        property_mls_id: "CA-2026-0429-8847".to_string(),
        tenant_id: "TENANT-8821".to_string(),
        issue_description: "Broken water heater — no hot water for 3 days".to_string(),
        urgency: 4,
        tenant_joy_impact: 7.5,
        cehi_impact: 1.8,
        estimated_cost: 1850.0,
        days_since_last_inspection: 267,
    };

    println!("📋 Processing maintenance request {}...", request.request_id);

    let result = maintenance_engine.process_maintenance_request(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ MAINTENANCE DEMO COMPLETE — PRIORITIZED & APPROVED            ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
