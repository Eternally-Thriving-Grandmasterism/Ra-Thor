//! examples/powrush_npc_v15_demo.rs
//! Demo showcasing Full Player-NPC Trading (Buy + Sell) + Harmony influence

use powrush::WorldSimulation;

fn main() {
    println!("\n=== Powrush v15.9 — Real Player-NPC Trading Demo ===\n");

    let mut sim = WorldSimulation::new();

    println!("Initial Economy Pool: {:.1} credits\n", sim.economy.current_pool());

    for tick in 0..12 {
        sim.tick(0.5);

        // Demonstrate explicit trading
        if tick == 4 {
            println!("\n--- Explicit Trading Example ---");
            let _ = sim.trade_with_npc(0, "Mercy Shard", 2, false); // Buy
        }

        if tick == 8 && sim.player.inventory.has("Mercy Shard") {
            println!("\n--- Player Sells to NPC ---");
            let _ = sim.trade_with_npc(0, "Mercy Shard", 1, true); // Sell
        }
    }

    println!("\nFinal Economy Pool: {:.1} credits", sim.economy.current_pool());
    println!("Final Player Inventory: {:?}", sim.player.inventory.items);
    println!("\n=== Demo Complete. Thunder locked. ===\n");
}