use patsagi_council_orchestrator::{get_active_councils, orchestrate_transmutation};

fn main() {
    println!("=== PATSAGi Council Orchestrator v0.1.0 ===");
    let councils = get_active_councils();
    for c in &councils {
        println!("Council #{}: {} — {}", c.id, c.role, c.specialty);
    }
    println!("\n{}", orchestrate_transmutation("Quantum Swarm + Powrush RBE"));
    println!("=== Councils in Eternal Session ===");
}