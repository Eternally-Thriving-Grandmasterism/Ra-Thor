//! Lease Renewal Demo — RREL v0.5.21
//! Demonstrates mercy-gated lease renewal with loyalty discounts and CEHI rewards

use real_estate_lattice::lease_renewal_engine::{LeaseRenewalEngine, LeaseRenewalRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           📝 RREL LEASE RENEWAL DEMO — v0.5.21                            ║");
    println!("║   Mercy-Gated • Quantum Swarm • Loyalty Discount System                  ║");
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
        renewal_id: "RENEW-2026-0429-9912".to_string(),
        property_mls_id: "ON-2026-0429-8821".to_string(),
        tenant_id: "TENANT-7719".to_string(),
        current_rent: 2850.0,
        years_as_tenant: 6,
        tenant_cehi: 9.1,
        payment_history_score: 9.4,
        requested_renewal_term_months: 24,
    };

    println!("📋 Processing lease renewal for {}...", request.tenant_id);

    let result = lease_engine.process_lease_renewal(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ LEASE RENEWAL DEMO COMPLETE — LOYAL TENANT REWARDED           ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
