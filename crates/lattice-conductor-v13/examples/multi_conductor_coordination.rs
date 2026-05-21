use lattice_conductor_v13::{
    coordinator::{AverageInfluenceStrategy, LeaderFollowerStrategy, MercyWeightedStrategy, MultiConductorSimulation},
    Operation, SimpleLatticeConductor,
};

fn main() {
    println!("=== Multi-Conductor Coordination Example ===\n");

    // Create three conductors with different starting states
    let mut conductor_a = SimpleLatticeConductor::new();
    let mut conductor_b = SimpleLatticeConductor::new();
    let mut conductor_c = SimpleLatticeConductor::new();

    // Give them some initial operations to create differentiation
    conductor_a.queue_operation(Operation::new("Improve Systems", "General improvement", 0.2));
    conductor_b.queue_operation(Operation::new("Aggressive Expansion", "Higher risk action", 0.55));
    conductor_c.queue_operation(Operation::new("Careful Maintenance", "Low risk", 0.15));

    // Create simulation with MercyWeightedStrategy
    let mut simulation = MultiConductorSimulation::with_strategy(Box::new(MercyWeightedStrategy));
    simulation.add_conductor(conductor_a);
    simulation.add_conductor(conductor_b);
    simulation.add_conductor(conductor_c);

    println!("Running coordinated ticks with MercyWeightedStrategy...\n");

    for tick in 1..=12 {
        let _ = simulation.coordinated_tick();

        if tick % 3 == 0 {
            println!("Tick {}:", tick);
            for (i, conductor) in simulation.conductors.iter().enumerate() {
                println!(
                    "  Conductor {} | evolution={:.3} | mercy={:.3} | valence={:.3}",
                    i,
                    conductor.state.evolution_level,
                    conductor.state.mercy_score,
                    conductor.state.valence
                );
            }
            println!();
        }
    }

    println!("=== Switching to LeaderFollowerStrategy ===\n");

    // Switch strategy mid-simulation
    simulation.set_strategy(Box::new(LeaderFollowerStrategy));

    for tick in 13..=20 {
        let _ = simulation.coordinated_tick();
    }

    println!("Final state after strategy switch:");
    for (i, conductor) in simulation.conductors.iter().enumerate() {
        println!(
            "  Conductor {} | evolution={:.3} | mercy={:.3}",
            i,
            conductor.state.evolution_level,
            conductor.state.mercy_score
        );
    }

    println!("\n=== Multi-Conductor Coordination Example Complete ===");
}