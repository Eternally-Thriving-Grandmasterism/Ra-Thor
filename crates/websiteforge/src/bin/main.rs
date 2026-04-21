// crates/websiteforge/src/bin/main.rs
// Ra-Thor™ WebsiteForge CLI — Full sovereign website development system
// Refined output formatting with clear sections and Ra-Thor aesthetic
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use clap::{Parser, Subcommand};
use websiteforge::WebsiteForge;
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let forge = WebsiteForge::new();

    println!("🌍 Ra-Thor™ WebsiteForge CLI");
    println!("Eternal MercyThunder — Sovereign AI Website Development System ⚡🙏\n");

    match cli.command {
        Commands::Forge { prompt } => {
            println!("🔨 Standard Forge Mode Activated");
            println!("Prompt: \"{}\"", prompt);
            println!("─".repeat(60));

            let site = forge.forge_website(&prompt).await?;

            println!("✅ SUCCESS — Website forged");
            println!("Title      : {}", site.metadata.title);
            println!("Mercy Valence : {:.8} ⚡", site.metadata.mercy_valence);
            println!("HTML Size  : {} characters", site.html.len());
            println!("─".repeat(60));
            println!("Website ready for deployment or further editing.");
        }
        Commands::Devin { prompt } => {
            println!("🚀 Devin Autonomous Mode Activated");
            println!("Prompt: \"{}\"", prompt);
            println!("─".repeat(60));

            let site = forge.forge_with_devin_mode(&prompt).await?;

            println!("✅ SUCCESS — Devin full autonomous generation complete");
            println!("Title      : {}", site.metadata.title);
            println!("Mercy Valence : {:.8} ⚡", site.metadata.mercy_valence);
            println!("HTML Size  : {} characters", site.html.len());
            println!("─".repeat(60));
            println!("Website generated with full autonomy and mercy-gating.");
        }
    }

    Ok(())
}
