//! # Powrush Single-Player Binary (v0.1.0)
//!
//! The world's first mercy-gated Resource-Based Economy (RBE) game — Single Player Mode.
//! Built on Ra-Thor + TOLC 7 Living Mercy Gates.
//!
//! Run with: cargo run -p powrush --bin powrush-singleplayer

use powrush::{PowrushCore, SimulationEngine, Faction};
use std::io::{self, Write};

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║           🌌  POWRUSH — Single Player Mode  🌌            ║");
    println!("║     The world's first mercy-gated RBE AGI game           ║");
    println!("║          Built on Ra-Thor + TOLC 7 Living Mercy Gates     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let mut engine = SimulationEngine::new();
    let mut core = PowrushCore::new();

    // Start with a default player
    core.game.add_player("You".to_string(), Faction::EternalCompassion);

    println!("Welcome, Seeker. The 7 Living Mercy Gates are now active.\n");

    loop {
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let cmd = input.trim().to_lowercase();

        match cmd.as_str() {
            "help" | "h" => {
                println!("\nCommands:");
                println!("  run          — Run one mercy-gated simulation cycle");
                println!("  status       — Show current world & player status");
                println!("  add <name>   — Add a new player (e.g. add Elara)");
                println!("  joy          — Show collective joy level");
                println!("  events       — List recent world events");
                println!("  quit         — Exit Powrush");
            }
            "run" => {
                match engine.run_multi_player_cycle() {
                    Ok(result) => println!("\n{}", result),
                    Err(e) => println!("\n⚠️  Mercy Gate violation: {}", e),
                }
            }
            "status" => {
                println!("\n{}", engine.get_world_summary());
                for player in &engine.game.players {
                    println!(
                        "  {} ({}) — Happiness: {:.1} | Joy: {:.1} | Ascension: {}",
                        player.name,
                        player.faction.name(),
                        player.happiness,
                        player.needs.joy,
                        player.ascension_level.name()
                    );
                }
            }
            cmd if cmd.starts_with("add ") => {
                let name = cmd[4..].trim();
                if !name.is_empty() {
                    engine.game.add_player(name.to_string(), Faction::Ambrosians);
                    println!("\n✨ {} has joined the world as an Ambrosian.", name);
                }
            }
            "joy" => {
                println!("\n🌸 Collective Joy: {:.1}/100", engine.collective_joy);
            }
            "events" => {
                if engine.world_events.is_empty() {
                    println!("\nNo major world events yet. Keep playing...");
                } else {
                    println!("\nRecent World Events:");
                    for event in engine.world_events.iter().rev().take(5) {
                        println!("  • {}", event.description);
                    }
                }
            }
            "quit" | "exit" | "q" => {
                println!("\nThank you for playing Powrush.\nMay mercy guide your path. ❤️🔥🌀");
                break;
            }
            _ => {
                println!("\nUnknown command. Type 'help' for options.");
            }
        }
    }
}
