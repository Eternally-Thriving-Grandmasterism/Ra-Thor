// crates/quantum-swarm-orchestrator/examples/one_organism_cycle_demo.rs
// Demonstration of ONE Organism cycle with LatticeConductorAdapter

use quantum_swarm_orchestrator::{
    LatticeConductorAdapter, OneOrganismContext, QuantumSwarmOrchestrator, Valence,
};

#[tokio::main]
async fn main() {
    println!("=== ONE Organism Cycle Demo ===\n");

    let mut orchestrator = QuantumSwarmOrchestrator::new();

    // Register the first adapter
    let lattice_adapter = Box::new(LatticeConductorAdapter::new());
    orchestrator.register_adapter(lattice_adapter);

    println!("Registered systems: {}", orchestrator.registered_system_count());

    // Create context (simulating PATSAGi + TOLC input)
    let context = OneOrganismContext {
        cycle_id: 1,
        tolc_order: 8,
        base_valence: Valence(0.99999997),
        patsagi_insight: Some("Focus on mercy-gated Real Estate conduction".to_string()),
    };

    // Run the organism cycle
    match orchestrator.run_one_organism_cycle(context).await {
        Ok(insight) => {
            println!("\nCycle completed successfully!");
            println!("Average valence: {:.8}", insight.average_valence);
            println!("Active systems: {:?}", insight.active_systems);
            println!("Recommended actions:");
            for action in insight.recommended_actions {
                println!("  - {}", action);
            }
        }
        Err(e) => {
            println!("Cycle failed: {}", e);
        }
    }

    println!("\n=== Demo Complete ===");
}