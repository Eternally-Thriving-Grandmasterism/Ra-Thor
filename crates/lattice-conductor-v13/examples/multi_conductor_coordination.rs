use lattice_conductor_v13::{
    coordinator::{AverageInfluenceStrategy, LeaderFollowerStrategy, MercyWeightedStrategy, MultiConductorSimulation},
    Operation, SimpleLatticeConductor,
};

fn main() {
    println!("=== Multi-Conductor Coordination with Adaptive Influence ===\n");

    let mut c1 = SimpleLatticeConductor::new();
    let mut c2 = SimpleLatticeConductor::new();
    let mut c3 = SimpleLatticeConductor::new();

    c1.queue_operation(Operation::new("Steady Progress", "Balanced", 0.25));
    c2.queue_operation(Operation::new("Bold Move", "Higher risk", 0.5));
    c3.queue_operation(Operation::new("Careful Work", "Low risk", 0.1));

    let mut sim = MultiConductorSimulation::with_strategy(Box::new(MercyWeightedStrategy));
    sim.add_conductor(c1);
    sim.add_conductor(c2);
    sim.add_conductor(c3);

    println!("Running with MercyWeightedStrategy + adaptive influence...\n");

    for tick in 1..=15 {
        let _ = sim.coordinated_tick();

        if tick % 5 == 0 {
            let avg_evolution = sim.conductors.iter().map(|c| c.state.evolution_level).sum::<f64>() / sim.conductors.len() as f64;
            let avg_evo_rate = sim.conductors.iter().map(|c| c.adaptive_params.evolution_rate).sum::<f64>() / sim.conductors.len() as f64;

            println!("Tick {} | Avg Evolution: {:.3} | Avg Evo Rate: {:.5}", tick, avg_evolution, avg_evo_rate);
        }
    }

    println!("\nSwitching to LeaderFollowerStrategy...");
    sim.set_strategy(Box::new(LeaderFollowerStrategy));

    for _ in 0..8 {
        let _ = sim.coordinated_tick();
    }

    println!("\nFinal adaptive parameters:");
    for (i, c) in sim.conductors.iter().enumerate() {
        println!("  Conductor {} | evolution_rate={:.5} | mercy_recovery={:.5}",
            i, c.adaptive_params.evolution_rate, c.adaptive_params.mercy_recovery_rate);
    }

    println!("\n=== Complete ===");
}