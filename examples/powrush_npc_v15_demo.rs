//! examples/powrush_npc_v15_demo.rs
//! Clean demo of WorldSimulation with stabilized Geometric Engine + RBE Economy + Visualization

use powrush::WorldSimulation;

fn main() {
    println!("\n=== Powrush v15 — Geometric Harmony + RBE Economy Demo ===\n");

    let mut sim = WorldSimulation::new();

    for _ in 0..12 {
        sim.tick(0.5);
    }

    println!("\nFinal Economy Pool: {:.1} credits", sim.current_economy_pool());
    println!("Final Geometric Harmony: {:.3}", sim.geometric_harmony_score);
    println!("\n=== Demo Complete — Thunder locked. ===\n");
}