//! Integration Tests for USA Pilot Module (v14.3)
//! Tests the full offer flow processing through UsaPilotModule

use real_estate_lattice::{UsaPilotModule, UsaOfferFlowReport};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use real_estate_lattice::usa_state_adapters::UsState;

fn create_pilot() -> UsaPilotModule {
    let mercy = MercyEngine::new();
    let swarm = QuantumSwarmOrchestrator::new();
    let governance = WorldGovernanceEngine::new();
    UsaPilotModule::new(mercy, swarm, governance)
}

#[tokio::test]
async fn test_usa_offer_flow_basic_california() {
    let mut pilot = create_pilot();
    let mut game = PowrushGame::new();

    let report: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::CA,
            "Standard home purchase in California with full disclosures",
            950_000.0,
            &mut game,
        )
        .await
        .unwrap();

    assert_eq!(report.state, "CA");
    println!("CA basic flow passed_regulatory: {}", report.passed_regulatory);
}

#[tokio::test]
async fn test_usa_offer_flow_florida_condo() {
    let mut pilot = create_pilot();
    let mut game = PowrushGame::new();

    let report: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::FL,
            "Condo purchase in Florida. Flood disclosure included but no milestone inspection mentioned.",
            720_000.0,
            &mut game,
        )
        .await
        .unwrap();

    println!("FL condo flow state_issues: {:?}", report.state_issues);
}

#[tokio::test]
async fn test_usa_offer_flow_high_value_texas() {
    let mut pilot = create_pilot();
    let mut game = PowrushGame::new();

    let report: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::TX,
            "Luxury home purchase, basic disclosures only",
            3_200_000.0,
            &mut game,
        )
        .await
        .unwrap();

    println!("TX high value federal_issues: {:?}", report.federal_issues);
}

#[tokio::test]
async fn test_usa_offer_flow_with_kickback_new_york() {
    let mut pilot = create_pilot();
    let mut game = PowrushGame::new();

    let report: UsaOfferFlowReport = pilot
        .process_usa_offer_flow(
            UsState::NY,
            "Purchase involving referral kickback arrangement",
            1_450_000.0,
            &mut game,
        )
        .await
        .unwrap();

    assert!(!report.passed_regulatory || report.federal_issues.iter().any(|i| i.contains("RESPA")));
    println!("NY kickback test passed_regulatory: {}", report.passed_regulatory);
}
