//! Tenant Screening Demo — RREL v0.5.21
//! Demonstrates mercy-gated, fair-housing-compliant tenant screening with ethical risk scoring

use real_estate_lattice::tenant_screening_engine::{TenantScreeningEngine, TenantApplication};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           👤 RREL TENANT SCREENING DEMO — v0.5.21                         ║");
    println!("║   Mercy-Gated • Quantum Swarm • Fair Housing Compliant                   ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut screening_engine = TenantScreeningEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let application = TenantApplication {
        application_id: "APP-2026-0429-7719".to_string(),
        property_mls_id: "ON-2026-0429-8821".to_string(),
        applicant_cehi: 8.9,
        credit_score: 712,
        income_to_rent_ratio: 3.8,
        rental_history_years: 6,
        eviction_history: 0,
        references_score: 9.2,
    };

    println!("🔍 Screening tenant application {}...", application.application_id);

    let result = screening_engine.screen_tenant_application(&application, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ TENANT SCREENING DEMO COMPLETE — FAIR HOUSING VERIFIED        ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
