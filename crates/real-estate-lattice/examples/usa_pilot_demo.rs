//! USA Pilot Demo — RREL v14.3
//!
//! Concrete demo of the implemented USA offer processing logic.
//!
//! Run with:
//!   cargo run --example usa_pilot_demo

use real_estate_lattice::{UsaPilotModule, UsaOfferFlowReport};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use real_estate_lattice::usa_state_adapters::UsState;

#[tokio::main]
async fn main() {
    println!("🇺🇸 RREL v14.3 USA Pilot Demo\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();

    let mut pilot = UsaPilotModule::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let mut game = PowrushGame::new();

    // California
    println!("=== California Offer ===");
    let ca: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::CA,
            "Residential purchase in Los Angeles. Full TILA + RESPA disclosures provided. No kickbacks.",
            1_250_000.0,
            &mut game,
        )
        .await
        .unwrap();
    print_report("California", &ca);

    // Florida
    println!("=== Florida Offer ===");
    let fl: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::FL,
            "Condo in Miami. Flood zone disclosure included.",
            875_000.0,
            &mut game,
        )
        .await
        .unwrap();
    print_report("Florida", &fl);

    // Texas
    println!("=== Texas Offer ===");
    let tx: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::TX,
            "Home in Austin. Property tax protest rights documented.",
            650_000.0,
            &mut game,
        )
        .await
        .unwrap();
    print_report("Texas", &tx);

    println!("✅ USA offer processing demo complete.");
}

fn print_report(state: &str, report: &UsaOfferFlowReport) {
    println!("State             : {}", state);
    println!("Passed            : {}", report.passed_regulatory);
    println!("Mercy             : {:.2}", report.mercy_valence);
    println!("Quantum           : {:.2}", report.quantum_consensus);
    if !report.federal_issues.is_empty() {
        println!("Federal Issues    : {:?}", report.federal_issues);
    }
    if !report.state_issues.is_empty() {
        println!("State Issues      : {:?}", report.state_issues);
    }
    println!("Summary           : {}", report.summary);
    println!();
}
