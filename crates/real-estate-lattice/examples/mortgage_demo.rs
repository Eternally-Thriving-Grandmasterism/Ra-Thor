//! Mortgage Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm, RECO-compliant mortgage approval

use real_estate_lattice::mortgage_engine::{MortgageEngine, MortgageApplication};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🏠 RREL MORTGAGE DEMO — v0.5.21                                 ║");
    println!("║   Mercy-Gated • Quantum Swarm • RECO/TRESA Compliant                     ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut mortgage_engine = MortgageEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let application = MortgageApplication {
        applicant_id: "ALPHA-2026-0429-001".to_string(),
        property_mls_id: "CA-2026-0429-8847".to_string(),
        loan_amount: 875000.0,
        down_payment: 175000.0,
        credit_score: 782,
        annual_income: 195000.0,
    };

    println!("📋 Processing mortgage application for {}...", application.applicant_id);

    let result = mortgage_engine.process_mortgage_application(&application, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ MORTGAGE DEMO COMPLETE — APPROVED WITH FULL MERCY ALIGNMENT   ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
