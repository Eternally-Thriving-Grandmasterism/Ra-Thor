//! Emergency Response & Disaster Recovery Demo — RREL v0.5.21
//! Demonstrates mercy-gated, life-saving disaster response and community recovery

use real_estate_lattice::emergency_response_disaster_recovery_engine::{EmergencyResponseDisasterRecoveryEngine, DisasterEvent};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🚨 RREL EMERGENCY RESPONSE & DISASTER RECOVERY DEMO — v0.5.21    ║");
    println!("║   Mercy-Gated • Quantum Swarm • Life-Saving Coordination                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut disaster_engine = EmergencyResponseDisasterRecoveryEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let event = DisasterEvent {
        event_id: "DISASTER-2026-0429-9912".to_string(),
        property_mls_id: "FL-2026-0429-8821".to_string(),
        disaster_type: "Hurricane".to_string(),
        severity_level: 9,
        affected_tenant_count: 87,
        community_cehi_score: 7.6,
        immediate_needs: vec![
            "Temporary housing for 62 families".to_string(),
            "Medical & mental health support".to_string(),
            "Food, water & emergency supplies".to_string(),
        ],
    };

    println!("🚨 Activating disaster response for {}...", event.event_id);

    let result = disaster_engine.activate_disaster_response(&event, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ DISASTER RESPONSE DEMO COMPLETE — MERCY IN ACTION             ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
