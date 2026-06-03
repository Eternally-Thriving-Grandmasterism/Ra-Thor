//! examples/powrush_mmo_demo.rs
//! Powrush MMO Demo — Runnable authoritative simulation using v16.0 core
//! Demonstrates PowrushMMOWorld, authoritative_tick, chunk regeneration, and basic session management

use powrush::simulation::{PowrushMMOWorld, WorldChunk};
use std::time::{Duration, Instant};

fn main() {
    println!("=== Powrush MMO Authoritative Demo (v16.0) ===");
    println!("Initializing PowrushMMOWorld...\n");

    let mut world: PowrushMMOWorld = PowrushMMOWorld::new();

    // Demo: Show initial chunks
    println!("Initial loaded chunks: {}", world.chunks.len());
    if let Some(chunk) = world.get_chunk((0, 0)) {
        println!("Chunk (0,0) mercy_essence: {:.1}", chunk.resources.get("mercy_essence").unwrap_or(&0.0));
    }

    println!("\nRunning 120 authoritative ticks (2 seconds @ 60 TPS simulation)...\n");

    let start = Instant::now();
    let target_ticks = 120;
    let tick_duration = Duration::from_millis(16); // ~60 TPS

    for tick in 0..target_ticks {
        world.authoritative_tick(0.016);

        if tick % 30 == 0 {
            println!("[Tick {:03}] Harmony: {:.3} | Economy: {:.1} cr | Chunks: {} | Active NPCs: {}",
                world.tick_count,
                world.geometric_harmony_score,
                world.economy.current_pool(),
                world.chunks.len(),
                world.active_npcs()
            );

            // Show resource regeneration progress on center chunk
            if let Some(chunk) = world.get_chunk((0, 0)) {
                let mercy = chunk.resources.get("mercy_essence").unwrap_or(&0.0);
                println!("   Chunk (0,0) mercy_essence: {:.1}", mercy);
            }
        }

        // Simulate very basic player input every 20 ticks
        if tick % 20 == 0 {
            // In real system this would come from PlayerSession input queue
            world.player.position.x = (world.player.position.x + 1.5) % 40.0;
        }

        std::thread::sleep(tick_duration);
    }

    let elapsed = start.elapsed();
    println!("\nDemo complete. Ran {} ticks in {:.2?}", target_ticks, elapsed);
    println!("Final tick count: {}", world.tick_count);
    println!("Final harmony: {:.3}", world.player.harmony);
    println!("\n=== Powrush MMO Demo Finished ===");
}