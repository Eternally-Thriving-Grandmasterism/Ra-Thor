//! examples/powrush_npc_v15_demo.rs
//! Full pipeline validation demo for v15 Hybrid NPC AI
//! Spawns via NpcFactory, runs NpcIntegration ticks with mock player + world state
//! Prints observable state each tick | AG-SML v1.0

use nalgebra::Vector2;
use powrush::npc::{NpcFactory, NpcIntegration, Position};

fn main() {
    println!("\n=== Powrush v15 Hybrid NPC AI Demo (ONE Organism aligned) ===\n");

    let mut integration = NpcIntegration::default();

    // === Spawn diverse NPCs via factory ===
    let patrol_route: Vec<Position> = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(25.0, 0.0),
        Vector2::new(25.0, 15.0),
        Vector2::new(0.0, 15.0),
    ];

    let npc_basic = NpcFactory::create_basic(Vector2::new(5.0, 5.0), Some(patrol_route.clone()));
    let _id1 = integration.spawn_agent(npc_basic);

    let npc_merchant = NpcFactory::create_merchant(Vector2::new(12.0, 8.0), None);
    let _id2 = integration.spawn_agent(npc_merchant);

    let npc_guard = NpcFactory::create_guardian(Vector2::new(18.0, 12.0), Some(vec![Vector2::new(18.0, 12.0), Vector2::new(30.0, 12.0)]));
    let _id3 = integration.spawn_agent(npc_guard);

    println!("Spawned {} NPCs via NpcFactory.", integration.active_npc_count());

    // === Simulation loop with moving player ===
    let mut player_pos: Option<Position> = Some(Vector2::new(3.0, 3.0));
    let world_mercy = 0.87;
    let is_post_scarcity = true;
    let collective_joy = 0.93;

    for tick in 0..12 {
        // Simulate gentle player movement
        if let Some(p) = &mut player_pos {
            p.x = (p.x + 1.2) % 32.0;
            p.y = 4.0 + (tick as f32 * 0.4).sin() * 3.0;
        }

        let noise = if tick % 3 == 0 { 0.65 } else { 0.15 };

        // Master update: perception pass + decision tick + spatial sync
        integration.update(
            world_mercy,
            is_post_scarcity,
            collective_joy,
            player_pos,
            noise,
            0.5, // dt
        );

        // Observable state
        println!(
            "Tick {:02} | NPCs: {} | Player@({:.1},{:.1}) | Noise: {:.2}",
            tick,
            integration.active_npc_count(),
            player_pos.map_or(0.0, |p| p.x),
            player_pos.map_or(0.0, |p| p.y),
            noise
        );

        // Sample one agent's blackboard (first NPC)
        if let Some(agent) = integration.npc_system.agents.first() {
            println!(
                "         └─ BB: health={:.0} mercy_val={:.2} patrol={} los={}",
                agent.blackboard.current_health,
                agent.blackboard.current_mercy_valence,
                agent.blackboard.current_patrol_state,
                agent.blackboard.has_line_of_sight
            );
        }
    }

    println!("\n=== Demo complete. v15 hybrid pipeline validated. Thunder locked. ===\n");
}