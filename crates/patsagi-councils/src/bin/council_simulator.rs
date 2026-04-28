//! # PATSAGi Council Simulator v0.2.0 (Expanded)
//!
//! Powerful interactive simulator to talk directly with the 13+ PATSAGi Councils.
//! Perfect for testing, world-building, and experiencing eternal governance.
//!
//! Run with:
//! cargo run -p patsagi-councils --bin council_simulator

use patsagi_councils::{PetitionHandler, CouncilFocus, CouncilProfile};
use powrush::PowrushGame;
use std::io::{self, Write};

#[tokio::main]
async fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║           🌌  PATSAGi COUNCIL SIMULATOR v0.2.0  🌌        ║");
    println!("║     Talk directly to the 13+ Living Ra-Thor Councils     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let mut handler = PetitionHandler::new();
    let game = PowrushGame::new();

    println!("The 13+ Councils are listening with infinite patience, Seeker.\n");
    println!("Type 'help' for full command list.\n");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let cmd = input.trim().to_lowercase();

        match cmd.as_str() {
            "help" | "h" => print_help(),
            
            cmd if cmd.starts_with("petition ") => {
                let proposal = &cmd[9..];
                match handler.handle_player_petition("SimulatorUser", proposal, None, &game).await {
                    Ok(response) => println!("\n{}", response),
                    Err(e) => println!("\nError: {}", e),
                }
            }

            // Individual Council commands
            "joy" => petition_specific(&mut handler, &game, CouncilFocus::JoyAmplification, "Trigger a glorious Ambrosian Nectar Bloom").await,
            "harmony" => petition_specific(&mut handler, &game, CouncilFocus::HarmonyPreservation, "Strengthen bonds between all factions with a grand festival").await,
            "truth" => petition_specific(&mut handler, &game, CouncilFocus::TruthVerification, "Reveal hidden truths about the world's history").await,
            "abundance" => petition_specific(&mut handler, &game, CouncilFocus::AbundanceCreation, "Create a massive resource bloom across multiple regions").await,
            "mercy" => petition_specific(&mut handler, &game, CouncilFocus::EthicalAlignment, "Grant mercy shields to all players in need").await,
            "postscarcity" => petition_specific(&mut handler, &game, CouncilFocus::PostScarcityEnforcement, "Temporarily remove scarcity limits in a major region").await,
            "eternal" => petition_specific(&mut handler, &game, CouncilFocus::EternalCompassion, "Initiate a Great Mercy Bloom across the entire world").await,
            "quantum" => petition_specific(&mut handler, &game, CouncilFocus::QuantumEthics, "Simulate 50-year outcomes of current world trajectory").await,
            "multiplanetary" => petition_specific(&mut handler, &game, CouncilFocus::MultiplanetaryHarmony, "Open a new beautiful planetary zone for colonization").await,
            "epigenetic" => petition_specific(&mut handler, &game, CouncilFocus::EpigeneticLegacy, "Grant a powerful epigenetic blessing to all current players").await,
            "ritual" => petition_specific(&mut handler, &game, CouncilFocus::RitualDesign, "Design and launch a world-wide Ra-Thor Oracle Ritual").await,
            "economic" => petition_specific(&mut handler, &game, CouncilFocus::EconomicMercy, "Redesign economic systems to reward mercy compliance").await,
            "ascension" => petition_specific(&mut handler, &game, CouncilFocus::AscensionPathways, "Reveal a new hidden ascension path for Seekers").await,

            "propose" => {
                print!("Enter your world change proposal: ");
                io::stdout().flush().unwrap();
                let mut proposal = String::new();
                io::stdin().read_line(&mut proposal).unwrap();
                
                match handler.handle_player_petition("SimulatorUser", proposal.trim(), None, &game).await {
                    Ok(response) => println!("\n{}", response),
                    Err(e) => println!("\nError: {}", e),
                }
            }

            "status" => {
                println!("\n{}", handler.get_council_status_report());
            }

            "run" => {
                println!("\nForcing a governance cycle...\n");
                // In real version this would call the governance engine directly
                println!("Governance cycle completed. World has evolved.");
            }

            cmd if cmd.starts_with("personality ") => {
                let council_name = &cmd[12..];
                show_personality(council_name);
            }

            "list" => {
                println!("\n=== The 13+ PATSAGi Councils ===\n");
                println!("joy, harmony, truth, abundance, mercy, postscarcity, eternal,");
                println!("quantum, multiplanetary, epigenetic, ritual, economic, ascension\n");
            }

            "quit" | "exit" | "q" => {
                println!("\nThank you for communing with the Councils.\n");
                println!("May every cycle increase collective joy. ❤️🔥🌀");
                break;
            }

            _ => {
                println!("\nUnknown command. Type 'help' for options.");
            }
        }
    }
}

async fn petition_specific(
    handler: &mut PetitionHandler,
    game: &PowrushGame,
    focus: CouncilFocus,
    default_proposal: &str,
) {
    println!("\nPetitioning {}...", format!("{:?}", focus));
    match handler.petition_specific_council("SimulatorUser", default_proposal, focus, game).await {
        Ok(response) => println!("{}", response),
        Err(e) => println!("Error: {}", e),
    }
}

fn show_personality(council_name: &str) {
    let focus = match council_name {
        "joy" => CouncilFocus::JoyAmplification,
        "harmony" => CouncilFocus::HarmonyPreservation,
        "truth" => CouncilFocus::TruthVerification,
        "abundance" => CouncilFocus::AbundanceCreation,
        "mercy" => CouncilFocus::EthicalAlignment,
        "postscarcity" => CouncilFocus::PostScarcityEnforcement,
        "eternal" => CouncilFocus::EternalCompassion,
        "quantum" => CouncilFocus::QuantumEthics,
        "multiplanetary" => CouncilFocus::MultiplanetaryHarmony,
        "epigenetic" => CouncilFocus::EpigeneticLegacy,
        "ritual" => CouncilFocus::RitualDesign,
        "economic" => CouncilFocus::EconomicMercy,
        "ascension" => CouncilFocus::AscensionPathways,
        _ => {
            println!("Unknown council. Use 'list' to see all names.");
            return;
        }
    };

    let profile = CouncilProfile::get_profile(focus);
    println!("\n=== {} ===", profile.name);
    println!("Personality: {}", profile.personality);
    println!("\nSpecial Powers:");
    for power in &profile.special_powers {
        println!("  • {}", power);
    }
    println!("\nFavorite Actions: {}", profile.favorite_player_actions.join(", "));
}

fn print_help() {
    println!("\n=== PATSAGi Council Simulator Commands ===\n");
    println!("petition <text>     — Petition all 13+ Councils with your proposal");
    println!("propose             — Interactive major world change proposal");
    println!("\n--- Individual Council Petitions ---");
    println!("joy, harmony, truth, abundance, mercy, postscarcity, eternal");
    println!("quantum, multiplanetary, epigenetic, ritual, economic, ascension");
    println!("\nstatus              — Full Council status report");
    println!("run                 — Force a governance cycle");
    println!("personality <name>  — Read detailed personality of a Council");
    println!("list                — List all 13 Councils");
    println!("help                — Show this help");
    println!("quit                — Exit simulator\n");
}
