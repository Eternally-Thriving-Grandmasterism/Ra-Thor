use lattice_conductor_v13::{Council, Operation, SimpleLatticeConductor, SimplePatsagiBridge};

fn main() {
    println!("=== PATSAGi-Influenced Conductor Demo ===\n");

    // Create a weighted PATSAGi council setup
    let bridge = SimplePatsagiBridge::with_councils(vec![
        Council { id: 1, name: "Mercy Council".to_string(), weight: 2.5, council_type: "Mercy".to_string() },
        Council { id: 2, name: "Truth Council".to_string(), weight: 1.5, council_type: "Truth".to_string() },
        Council { id: 3, name: "Evolution Council".to_string(), weight: 1.0, council_type: "Evolution".to_string() },
    ]);

    let mut conductor = SimpleLatticeConductor::new()
        .with_patsagi_bridge(Box::new(bridge));

    // Queue operations with varying harm levels
    let operations = vec![
        Operation::new("Collaborate on Project", "Positive action", 0.25),
        Operation::new("Complex Ethical Choice", "Medium harm potential", 0.55),
        Operation::new("High Risk Decision", "Significant potential harm", 0.82),
    ];

    for op in operations {
        let approved = conductor.validate_mercy(&op);
        println!(
            "Operation: '{}' | Harm: {:.2} | PATSAGi Approved: {}",
            op.name, op.potential_harm, approved
        );

        if approved {
            conductor.queue_operation(op);
        }
    }

    println!("\nRunning 8 ticks with PATSAGi-influenced mercy...\n");

    for i in 1..=8 {
        let _ = conductor.tick();
        let state = conductor.get_geometric_state();
        println!(
            "Tick {} | valence={:.3} | evolution={:.3} | swarm_coherence={:.3}",
            i, state.valence, state.evolution_level, conductor.quantum_swarm.coherence
        );
    }

    println!("\nFinal evolution level: {:.3}", conductor.state.evolution_level);
    println!("Mercy violations recorded: {}", conductor.get_mercy_violations().len());
    println!("=== Demo Complete ===");
}