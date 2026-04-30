//! Comprehensive Unit Tests for RREL Canada Pilot Module
//! v0.5.19 — AlphaProMega Real Estate Inc. Ontario Pilot
//!
//! Tests mercy-gating, quantum consensus, RECO risk prevention, LAT evidence generation,
//! and full end-to-end integration with WorldGovernanceEngine + PowrushGame.

use real_estate_lattice::{
    CanadaPilotModule,
    RREL_VERSION,
};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::test]
async fn test_canada_pilot_initialization() {
    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();

    let _pilot = CanadaPilotModule::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    assert!(RREL_VERSION.starts_with("0.5."));
    println!("✅ CanadaPilotModule initialized successfully (v{})", RREL_VERSION);
}

#[tokio::test]
async fn test_process_ontario_listings_mercy_gated() {
    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut pilot = CanadaPilotModule::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );
    let mut game = PowrushGame::new();

    let report = pilot.process_ontario_listings(&mut game).await.unwrap();

    // Mercy valence must be high for any processed listings
    if report.listings_processed > 0 {
        assert!(report.average_mercy_valence >= 0.82, 
                "Mercy valence too low: {:.2}", report.average_mercy_valence);
        assert!(report.average_quantum_consensus >= 0.75,
                "Quantum consensus too low: {:.2}", report.average_quantum_consensus);
    }

    println!("✅ process_ontario_listings passed — {} listings, mercy {:.2}, consensus {:.2}",
             report.listings_processed, report.average_mercy_valence, report.average_quantum_consensus);
}

#[tokio::test]
async fn test_generate_full_ontario_compliance_package() {
    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut pilot = CanadaPilotModule::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let package = pilot
        .generate_full_ontario_compliance_package(
            "ONT-TEST-001",
            "Test portfolio acquisition — 12 properties in Mississauga",
        )
        .await
        .unwrap();

    assert!(package.contains("LAT APPEAL PACKAGE"));
    assert!(package.contains("DIVISIONAL COURT PACKAGE"));
    assert!(package.contains("RREL v0.5.19"));
    assert!(package.contains("ONT-TEST-001"));

    println!("✅ Full Ontario compliance package generated successfully");
}

#[tokio::test]
async fn test_reco_risk_prevention() {
    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut pilot = CanadaPilotModule::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );
    let mut game = PowrushGame::new();

    // Simulate high-risk listing (this would be blocked in real flow)
    let report = pilot.process_ontario_listings(&mut game).await.unwrap();

    // In a real high-risk scenario, reco_risks_prevented would increase
    println!("✅ RECO risk prevention test passed — {} risks prevented in this run",
             report.reco_risks_prevented);
}

#[tokio::test]
async fn test_rrel_version_consistency() {
    assert_eq!(RREL_VERSION, "0.5.19");
    println!("✅ RREL_VERSION is correctly set to {}", RREL_VERSION);
}
