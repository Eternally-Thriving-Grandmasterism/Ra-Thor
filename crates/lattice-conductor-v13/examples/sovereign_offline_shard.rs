use lattice_conductor_v13::{Operation, SimpleLatticeConductor};

fn main() {
    println!("=== Sovereign Offline Shard Demo ===\n");

    let path = "/tmp/sovereign_conductor.json";

    // Try to load existing state (offline persistence)
    let mut conductor = if let Ok(loaded) = SimpleLatticeConductor::load_from_file(path) {
        println!("Loaded existing sovereign state from disk.");
        loaded
    } else {
        println!("No previous state found. Starting fresh sovereign conductor.");
        let mut c = SimpleLatticeConductor::new();
        c.register_council(1, "Sovereign Core");
        c
    };

    // Queue some operations
    conductor.queue_operation(Operation::new("Maintain Infrastructure", "Keep systems running", 0.15));
    conductor.queue_operation(Operation::new("Self Optimize", "Improve local processes", 0.25));

    // Run several ticks in offline mode
    for i in 1..=5 {
        let _ = conductor.tick();
        println!(
            "Tick {} | valence={:.3} | evolution={:.3} | swarm_branches={}",
            i,
            conductor.state.valence,
            conductor.state.evolution_level,
            conductor.quantum_swarm.active_branches
        );
    }

    // Save state for next offline run
    if let Err(e) = conductor.save_to_file(path) {
        eprintln!("Failed to save state: {}", e);
    } else {
        println!("\nSovereign state saved to {}", path);
    }

    println!("\nEvents recorded this session: {}", conductor.get_events().len());
    println!("=== Offline Shard Demo Complete ===");
}