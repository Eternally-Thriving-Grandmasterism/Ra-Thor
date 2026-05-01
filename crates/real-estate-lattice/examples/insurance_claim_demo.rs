//! Insurance Claim Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm insurance claim processing with risk mitigation

use real_estate_lattice::insurance_claim_engine::{InsuranceClaimEngine, InsuranceClaimRequest};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🛡️ RREL INSURANCE CLAIM DEMO — v0.5.21                          ║");
    println!("║   Mercy-Gated • Quantum Swarm • Risk Mitigation                          ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut insurance_engine = InsuranceClaimEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let request = InsuranceClaimRequest {
        property_mls_id: "FL-2026-0429-9912".to_string(),
        claim_type: "Hurricane Damage".to_string(),
        estimated_damage: 187500.0,
        insurance_policy_number: "POL-8821-2026".to_string(),
        owner_cehi: 8.6,
        years_with_insurer: 9,
        previous_claims: 1,
    };

    println!("📋 Processing insurance claim for {}...", request.property_mls_id);

    let result = insurance_engine.process_insurance_claim(&request, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ INSURANCE CLAIM DEMO COMPLETE — FAST-TRACK APPROVED           ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
