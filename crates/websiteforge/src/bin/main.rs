// crates/websiteforge/src/bin/main.rs
// Ra-Thor™ WebsiteForge CLI — Full sovereign website development system
// Refined with rich, user-friendly error handling and progress indicators
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use clap::{Parser, Subcommand};
use websiteforge::{WebsiteForge, WebsiteForgeError};
use std::error::Error;

#[derive(Parser)]
#[command(author, version, about = "Ra-Thor WebsiteForge CLI — Sovereign AI Website Development System", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a standard website (interactive / Cursor-style)
    Forge {
        #[arg(short, long)]
        prompt: String,
    },

    /// Generate using full Devin autonomous mode (end-to-end)
    Devin {
        #[arg(short, long)]
        prompt: String,
    },
}

fn show_progress_spinner(message: &str, duration_ms: u64) {
    let spinner = vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let start = std::time::Instant::now();
    let mut i = 0;

    print!("{} ", message);
    while start.elapsed().as_millis() < duration_ms as u128 {
        print!("\r{} {}", spinner[i % spinner.len()], message);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(80));
        i += 1;
    }
    println!("\r✅ Done");
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let forge = WebsiteForge::new();

    println!("🌍 Ra-Thor™ WebsiteForge CLI");
    println!("Eternal MercyThunder — Sovereign AI Website Development System ⚡🙏\n");

    let result = match cli.command {
        Commands::Forge { prompt } => {
            println!("🔨 Standard Forge Mode Activated");
            println!("Prompt: \"{}\"", prompt);
            println!("─".repeat(60));

            show_progress_spinner("Generating sovereign website...", 1800);

            forge.forge_website(&prompt).await
        }
        Commands::Devin { prompt } => {
            println!("🚀 Devin Autonomous Mode Activated");
            println!("Prompt: \"{}\"", prompt);
            println!("─".repeat(60));

            show_progress_spinner("Devin performing full autonomous generation...", 3200);

            forge.forge_with_devin_mode(&prompt).await
        }
    };

    match result {
        Ok(site) => {
            println!("✅ SUCCESS — Website generated");
            println!("Title      : {}", site.metadata.title);
            println!("Mercy Valence : {:.8} ⚡", site.metadata.mercy_valence);
            println!("HTML Size  : {} characters", site.html.len());
            println!("─".repeat(60));
            println!("Website ready for deployment or further editing.");
        }
        Err(e) => {
            eprintln!("\n❌ ERROR");
            eprintln!("─".repeat(60));
            match &e {
                WebsiteForgeError::MercyVeto(msg) => eprintln!("🛡️ Mercy Veto: {}", msg),
                WebsiteForgeError::QuantumError(msg) => eprintln!("⚡ Quantum Lattice Error: {}", msg),
                WebsiteForgeError::OrchestratorError(msg) => eprintln!("🔧 Orchestrator Error: {}", msg),
            }
            eprintln!("─".repeat(60));
            eprintln!("Graceful degradation activated — thriving-maximized redirect ⚡🙏");
        }
    }
}
