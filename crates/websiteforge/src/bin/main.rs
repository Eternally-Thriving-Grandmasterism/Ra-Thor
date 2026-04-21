// crates/websiteforge/src/bin/main.rs
// Ra-Thor™ WebsiteForge CLI — Full sovereign website development system
// Supports standard forge, Devin mode, and improved UX/help
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

    println!("🌍 Ra-Thor™ WebsiteForge CLI — Sovereign AI Website Development System");
    println!("Lightning already in motion. ⚡🙏\n");

    match cli.command {
        Commands::Forge { prompt } => {
            println!("🔨 Forging website with standard mode...");
            let site = forge.forge_website(&prompt).await?;
            println!("✅ Website forged successfully: {}", site.metadata.title);
            println!("Mercy valence: {:.8}", site.metadata.mercy_valence);
            println!("\nHTML preview: {} characters", site.html.len());
        }
        Commands::Devin { prompt } => {
            println!("🚀 Devin Mode activated — full autonomous generation...");
            let site = forge.forge_with_devin_mode(&prompt).await?;
            println!("✅ Devin-generated website: {}", site.metadata.title);
            println!("Mercy valence: {:.8}", site.metadata.mercy_valence);
            println!("\nHTML preview: {} characters", site.html.len());
        }
    }

    Ok(())
}
