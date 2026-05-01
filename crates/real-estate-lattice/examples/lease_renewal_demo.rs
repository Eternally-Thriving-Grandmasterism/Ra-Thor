//! Lease Renewal Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm lease renewal with loyalty-based discounts

use real_estate_lattice::lease_renewal::{LeaseRenewalEngine, LeaseRenewalRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           📜 RREL LEASE RENEWAL DEMO — v0.5.21                            ║");
    println!("║   Mercy-Gated • Quantum Swarm • Loyalty-Based Discounts                  ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut lease_engine = LeaseRenewalEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = LeaseRenewalRequest {
        tenant_id: "TENANT-8821".to_string(),
        property_mls_id: "CA-2026-0429-8847".to_string(),
        current_rent: 2850.0,
        lease_end_date: chrono::Utc::now() + chrono::Duration::days(45),
        tenant_cehi: 8.7,
        years_as_tenant: 6,
        payment_history_score: 0.97,
    };

    println!("📋 Processing lease renewal for {}...", request.tenant_id);

    let result = lease_engine.process_lease_renewal(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ LEASE RENEWAL DEMO COMPLETE — DISCOUNT APPLIED                ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
