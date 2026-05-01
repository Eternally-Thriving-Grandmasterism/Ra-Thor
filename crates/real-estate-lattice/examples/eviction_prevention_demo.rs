//! Eviction Prevention Demo — RREL v0.5.21
//! Demonstrates mercy-gated, quantum-swarm eviction prevention with mercy-first intervention

use real_estate_lattice::eviction_prevention::{EvictionPreventionEngine, EvictionRiskCase};
use patsagi_councils::WorldGovernanceEngine;
use powrush::PowrushGame;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🛡️ RREL EVICTION PREVENTION DEMO — v0.5.21                     ║");
    println!("║   Mercy-Gated • Quantum Swarm • Mercy-First Intervention                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmOrchestrator::new();
    let world_governance = WorldGovernanceEngine::new();
    let mut game = PowrushGame::new();

    let mut eviction_engine = EvictionPreventionEngine::new(
        mercy_engine,
        quantum_swarm,
        world_governance,
    );

    let case = EvictionRiskCase {
        tenant_id: "TENANT-8821".to_string(),
        property_mls_id: "CA-2026-0429-8847".to_string(),
        months_behind: 3,
        total_arrears: 6850.0,
        tenant_cehi: 7.4,
        previous_mercy_interventions: 1,
        hardship_reason: "Job loss due to medical emergency".to_string(),
    };

    println!("📋 Processing eviction prevention case for {}...", case.tenant_id);

    let result = eviction_engine.prevent_eviction(&case, &mut game).await?;

    println!("\n{}", result);

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║           ✅ EVICTION PREVENTION DEMO COMPLETE — MERCY INTERVENTION SUCCESS║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
