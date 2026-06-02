//! examples/powrush_npc_v15_demo.rs
//! Demo showcasing Per-NPC Harmony, Expanded Economy Items, and Crafting Recipes

use powrush::WorldSimulation;

fn main() {
    println!("\n=== Powrush v15.2 — Per-NPC Harmony + Economy + Crafting Demo ===\n");

    let mut sim = WorldSimulation::new();

    for _ in 0..14 {
        sim.tick(0.5);
    }

    println!("\nFinal Global Harmony: {:.3}", sim.geometric_harmony_score);
    println!("Final Economy Pool: {:.1} credits", sim.economy.current_pool());
    println!("\n=== Demo Complete. Thunder locked. ===\n");
}