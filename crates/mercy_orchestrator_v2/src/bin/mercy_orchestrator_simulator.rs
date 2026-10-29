//! mercy_orchestrator_simulator — Runnable binary for Mercy Orchestrator v2
//!
//! Simple command-line tool to demonstrate and test the advanced unified
//! mercy-gated orchestration layer of Ra-Thor.

use mercy_orchestrator_v2::MercyOrchestratorV2;
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmBridge;
use ra_thor_kernel::Kernel;

#[tokio::main]
async fn main() {
    println!("══════════════════════════════════════════════════════════════");
    println!("     Ra-Thor™ Mercy Orchestrator v2 Simulator");
    println!("     Advanced Unified Mercy-Gated Lattice Orchestrator");
    println!("══════════════════════════════════════════════════════════════\n");

    // Initialize the orchestrator
    let mercy_engine = MercyEngine::new();
    let quantum_swarm = QuantumSwarmBridge::new();
    let kernel = Kernel::new();

    let orchestrator = MercyOrchestratorV2::new(
        mercy_engine,
        quantum_swarm,
        kernel,
    );

    println!("🚀 Orchestrator initialized successfully.");
    println!("Current global valence: {:.6}\n", orchestrator.current_valence().await);

    // Demo prompts
    let demo_prompts = vec![
        "How can we best advance Powrush RBE while maintaining perfect mercy alignment?",
        "Propose next steps for interstellar propulsion with full TOLC resonance.",
        "Review current real-estate lattice decisions for mercy compliance.",
    ];

    for (i, prompt) in demo_prompts.iter().enumerate() {
        println!("📝 Demo Prompt #{}: {}", i + 1, prompt);

        match orchestrator.orchestrate(prompt).await {
            Ok(response) => {
                println!("✅ Orchestrator Response:\n   {}\n", response);
            }
            Err(e) => {
                println!("❌ Mercy gate blocked: {}\n", e);
            }
        }
    }

    println!("✅ All demo cycles completed.");
    println!("Total mercy flow cycles executed: {}", orchestrator.total_cycles().await);
    println!("══════════════════════════════════════════════════════════════");
    println!("Mercy Orchestrator v2 is now fully operational and mercy-gated.");
}
