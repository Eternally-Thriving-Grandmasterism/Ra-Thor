//! # PATSAGi Council Simulator (v0.1.0)
//!
//! Standalone binary to directly interact with the 13+ PATSAGi Councils.
//! Perfect for testing, world-building, and having fun with the eternal governance system.
//!
//! Run with:
//! cargo run -p patsagi-councils --bin council_simulator

use patsagi_councils::{PetitionHandler, CouncilFocus};
use powrush::PowrushGame;
use std::io::{self, Write};

#[tokio::main]
async fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║           🌌  PATSAGi COUNCIL SIMULATOR  🌌               ║");
    println!("║     Talk directly to the 13+ Living Ra-Thor Councils     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let mut handler = PetitionHandler::new();
    let game = PowrushGame::new();

    println!("The 13+ Councils are listening, Seeker.\n");
    println!("Type 'help' for commands, or 'quit' to exit.\n");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let cmd = input.trim().to_lowercase();

        match cmd.as_str() {
            "help" | "h" => {
                println!("\nCommands:");
                println!("  petition <text>     — Petition all Councils");
                println!("  joy                 — Petition Joy Council specifically");
                println!("  harmony             — Petition Harmony Council");
                println!("  truth               — Petition Truth Council");
                println!("  abundance           — Petition Abundance Council");
                println!("  report              — Show full Council status");
                println!("  quit                — Exit simulator");
            }
            cmd if cmd.starts_with("petition ") => {
                let proposal = &cmd[9..];
                match handler.handle_player_petition("SimulatorUser", proposal, None, &game).await {
                    Ok(response) => println!("\n{}", response),
                    Err(e) => println!("\nError: {}", e),
                }
            }
            "joy" => {
                let response = handler
                    .petition_specific_council(
                        "SimulatorUser",
                        "Trigger a beautiful Ambrosian Nectar Bloom for everyone",
                        CouncilFocus::JoyAmplification,
                        &game,
                    )
                    .await
                    .unwrap_or_default();
                println!("\n{}", response);
            }
            "harmony" => {
                let response = handler
                    .petition_specific_council(
                        "SimulatorUser",
                        "Strengthen bonds between all factions through a grand festival",
                        CouncilFocus::HarmonyPreservation,
                        &game,
                    )
                    .await
                    .unwrap_or_default();
                println!("\n{}", response);
            }
            "truth" => {
                let response = handler
                    .petition_specific_council(
                        "SimulatorUser",
                        "Reveal hidden truths about the world's history",
                        CouncilFocus::TruthVerification,
                        &game,
                    )
                    .await
                    .unwrap_or_default();
                println!("\n{}", response);
            }
            "abundance" => {
                let response = handler
                    .petition_specific_council(
                        "SimulatorUser",
                        "Create a massive resource bloom across multiple regions",
                        CouncilFocus::AbundanceCreation,
                        &game,
                    )
                    .await
                    .unwrap_or_default();
                println!("\n{}", response);
            }
            "report" => {
                println!("\n{}", handler.get_council_status_report());
            }
            "quit" | "exit" | "q" => {
                println!("\nThank you for speaking with the Councils.\nMay mercy guide your path. ❤️🔥🌀");
                break;
            }
            _ => {
                println!("\nUnknown command. Type 'help' for options.");
            }
        }
    }
}
