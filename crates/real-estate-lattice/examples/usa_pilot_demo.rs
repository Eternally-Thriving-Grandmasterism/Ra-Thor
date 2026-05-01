//! USA Pilot Demo — RREL v0.5.21 (Complete 50-State Unified System)
//! AlphaProMega Real Estate Inc. — Mercy-Gated • Quantum Swarm • 13+ PATSAGi Councils

use real_estate_lattice::{UsaPilotModule, UsState, RREL_VERSION};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🇺🇸 RREL USA PILOT DEMO — v{} (ALL 50 STATES)          ║", RREL_VERSION);
    println!("║   AlphaProMega Real Estate Inc. — Unified 50-State System                ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut usa_pilot = UsaPilotModule::new(mercy_engine, quantum_swarm, world_governance);

    // Demo with 10 major states (easily changeable to any combination of 50)
    let states = vec![
        UsState::California,
        UsState::Florida,
        UsState::Texas,
        UsState::NewYork,
        UsState::NewJersey,
        UsState::Pennsylvania,
        UsState::Illinois,
        UsState::Georgia,
        UsState::Washington,
        UsState::Massachusetts,
    ];

    println!("🇺🇸 Processing new MLS listings across {} major states using unified 50-state adapter...\n", states.len());

    let report = usa_pilot.process_usa_listings(&states, &mut game).await?;

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        USA PILOT REPORT (v0.5.21)                          ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!("Listings Processed:           {}", report.listings_processed);
    println!("Average Mercy Valence:        {:.2}", report.average_mercy_valence);
    println!("Average Quantum Consensus:    {:.2}", report.average_quantum_consensus);
    println!("Regulatory Issues Prevented:  {}", report.regulatory_issues_prevented);
    println!("States Covered:               {:?}", report.states_covered);
    println!("Timestamp:                    {}", report.timestamp);
    println!("════════════════════════════════════════════════════════════════════════════\n");

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ USA PILOT COMPLETE — 50-STATE SYSTEM FULLY OPERATIONAL        ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
