//! Mortgage Processing Demo — RREL v0.5.21
//! Demonstrates mercy-gated, ethical mortgage approval with CEHI-weighted scoring

use real_estate_lattice::mortgage_processing_engine::{MortgageProcessingEngine, MortgageApplication};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🏦 RREL MORTGAGE PROCESSING DEMO — v0.5.21                      ║");
    println!("║   Mercy-Gated • Quantum Swarm • Ethical Home Financing                   ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut mortgage_engine = MortgageProcessingEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let application = MortgageApplication {
        application_id: "MORT-2026-0429-9912".to_string(),
        property_mls_id: "ON-2026-0429-8821".to_string(),
        applicant_cehi: 8.7,
        credit_score: 742,
        debt_to_income_ratio: 0.31,
        down_payment_percentage: 0.22,
        loan_amount: 685000.0,
        years_in_current_job: 5,
    };

    println!("🏦 Processing mortgage application {}...", application.application_id);

    let result = mortgage_engine.process_mortgage_application(&application, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ MORTGAGE PROCESSING DEMO COMPLETE — PRE-APPROVED & MERCY VERIFIED ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
