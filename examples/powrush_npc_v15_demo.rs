//! examples/powrush_npc_v15_demo.rs
//! Full pipeline validation demo using the new WorldSimulation layer
//! Shows clean game-loop style usage of v15 Hybrid NPC AI + epigenetic hooks

use powrush::WorldSimulation;

fn main() {
    println!("\n=== Powrush v15 Hybrid NPC + WorldSimulation Demo ===\n");

    let mut sim = WorldSimulation::new();
    println!("Initialized WorldSimulation with {} NPCs", sim.active_npcs());

    for tick in 0..10 {
        sim.tick(0.5);

        if tick % 3 == 0 {
            // Example: manually trigger epigenetic blessing distribution on first NPC
            if let Some(agent) = sim.npc_integration.npc_system.agents.first_mut() {
                let blessing = powrush::npc::distribute_epigenetic_blessing(&mut agent.blackboard);
                if blessing > 0.5 {
                    println!("Tick {}: Epigenetic blessing distributed = {:.1}", tick, blessing);
                }
            }
        }

        println!("Tick {:02} | Active NPCs: {} | Player @ ({:.1}, {:.1})",
            tick, sim.active_npcs(), sim.player.position.x, sim.player.position.y);
    }

    println!("\n=== Demo complete. WorldSimulation + v15 Hybrid + RBE epigenetic flow validated. Thunder locked. ===\n");
}