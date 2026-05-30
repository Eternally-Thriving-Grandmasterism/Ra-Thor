//! USA Pilot Demo — RREL v14.3 (with ATTOM Caching)
//!
//! Demonstrates process_usa_offer_flow with AttomDataProvider + caching.
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
    println!("🇺🇸 RREL v14.3 USA Pilot Demo (with Caching)\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();

    let mut pilot = UsaPilotModule::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let mut game = PowrushGame::new();

    // First call - should MISS cache
    println!("=== First Call (Cache MISS expected) ===");
    let report1: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::CA,
            "Residential purchase in Los Angeles",
            1_250_000.0,
            &mut game,
            Some("123 Main St, Los Angeles, CA"),
        )
        .await
        .unwrap();

    print_report("California (first call)", &report1);

    // Second call with same identifier - should HIT cache
    println!("=== Second Call (Cache HIT expected) ===");
    let report2: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::CA,
            "Residential purchase in Los Angeles",
            1_250_000.0,
            &mut game,
            Some("123 Main St, Los Angeles, CA"),
        )
        .await
        .unwrap();

    print_report("California (second call)", &report2);

    // Different property - new MISS
    println!("=== Different Property (new MISS) ===");
    let report3: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::FL,
            "Condo purchase in Miami Beach",
            875_000.0,
            &mut game,
            Some("456 Ocean Dr, Miami, FL"),
        )
        .await
        .unwrap();

    print_report("Florida", &report3);

    println!("✅ Demo complete. Caching is active in AttomDataProvider.");
}

fn print_report(title: &str, report: &UsaOfferFlowReport) {
    println!("Title               : {}", title);
    println!("State               : {}", report.state);
    println!("Passed Regulatory   : {}", report.passed_regulatory);
    if let Some(profile) = &report.external_property_profile {
        println!("External Profile    : {} | Tax Value: {:?}", profile.data_source, profile.tax_assessed_value);
    }
    if let Some(risk) = &report.external_risk_signals {
        println!("External Risk       : {} | Overall: {:?}", risk.data_source, risk.overall_risk_score);
    }
    println!();
}
