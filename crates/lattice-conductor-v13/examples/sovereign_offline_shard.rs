use lattice_conductor_v13::{Operation, SimpleLatticeConductor};

fn main() {
    println!("=== Sovereign Offline Shard - Multi-Session Simulation ===\n");

    let path = "/tmp/sovereign_multi_session.json";

    // Session 1
    println!("--- Session 1 ---");
    let mut conductor = if let Ok(loaded) = SimpleLatticeConductor::load_from_file(path) {
        println!("Loaded previous sovereign state.");
        loaded
    } else {
        let mut c = SimpleLatticeConductor::new();
        c.register_council(1, "Sovereign Core");
        c
    };

    conductor.queue_operation(Operation::new("Maintain Systems", "Keep running", 0.2));
    conductor.queue_operation(Operation::new("Self Optimize", "Improve locally", 0.3));

    for i in 1..=6 {
        let _ = conductor.tick();
    }

    println!("After Session 1: evolution={:.3}, valence={:.3}", 
             conductor.state.evolution_level, conductor.state.valence);

    conductor.save_to_file(path).unwrap();

    // Session 2 (simulating restart)
    println!("\n--- Session 2 (after restart) ---");
    let mut conductor2 = SimpleLatticeConductor::load_from_file(path).unwrap();

    conductor2.queue_operation(Operation::new("Continue Evolution", "Build on previous", 0.25));

    for i in 1..=6 {
        let _ = conductor2.tick();
    }

    println!("After Session 2: evolution={:.3}, valence={:.3}", 
             conductor2.state.evolution_level, conductor2.state.valence);

    conductor2.save_to_file(path).unwrap();
    println!("\nState successfully persisted across sessions.");
    println!("=== Multi-Session Simulation Complete ===");
}