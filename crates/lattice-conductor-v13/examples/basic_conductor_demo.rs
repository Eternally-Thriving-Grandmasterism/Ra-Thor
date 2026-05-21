use lattice_conductor_v13::{Operation, SimpleLatticeConductor};

fn main() {
    println!("=== Lattice Conductor v13 Demo ===\n");

    let mut conductor = SimpleLatticeConductor::new();

    // Register some councils
    conductor.register_council(1, "Truth Council");
    conductor.register_council(2, "Mercy Council");

    println!("Registered councils: {:?}", conductor.get_registered_patsagi_councils());

    // Queue some operations
    conductor.queue_operation(Operation::new("Support Community", "Help people", 0.2));
    conductor.queue_operation(Operation::new("Exploit Resources", "Harmful action", 0.85));
    conductor.queue_operation(Operation::new("Share Knowledge", "Educational", 0.1));

    println!("\nRunning 3 ticks...\n");

    for i in 1..=3 {
        let _ = conductor.tick();
        let state = conductor.get_geometric_state();
        println!(
            "Tick {}: valence={:.3}, mercy_score={:.3}, tolc_alignment={:.3}",
            i, state.valence, state.mercy_score, state.tolc_alignment
        );
    }

    println!("\nMercy violations recorded: {}", conductor.get_mercy_violations().len());
    println!("Operations processed: {}", conductor.metrics.operations_processed);

    println!("\n=== Demo Complete ===");
}