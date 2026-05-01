//! Tenant Rights & Habitability Demo — RREL v0.5.21
//! Demonstrates mercy-gated tenant rights enforcement and habitability protection

use real_estate_lattice::tenant_rights_habitability_engine::{TenantRightsHabitabilityEngine, HabitabilityInspection};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🏠 RREL TENANT RIGHTS & HABITABILITY DEMO — v0.5.21             ║");
    println!("║   Mercy-Gated • Quantum Swarm • Tenant Protection & Habitability         ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut habitability_engine = TenantRightsHabitabilityEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let inspection = HabitabilityInspection {
        inspection_id: "HAB-2026-0429-9912".to_string(),
        property_mls_id: "ON-2026-0429-8821".to_string(),
        tenant_id: "TENANT-7719".to_string(),
        issues_found: vec![
            "No working heat for 12 days".to_string(),
            "Mold in bathroom and bedroom".to_string(),
            "Broken smoke detector".to_string(),
        ],
        tenant_cehi: 7.8,
        days_since_last_inspection: 312,
        landlord_response_time_days: 18,
    };

    println!("🏠 Processing habitability inspection {}...", inspection.inspection_id);

    let result = habitability_engine.process_habitability_inspection(&inspection, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ TENANT RIGHTS DEMO COMPLETE — HABITABILITY ENFORCED           ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
