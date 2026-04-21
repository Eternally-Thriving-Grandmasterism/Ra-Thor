// crates/websiteforge/src/bin/main.rs
// Ra-Thor™ WebsiteForge CLI — Full sovereign website development system
// Now supports Standard, Devin, Grok, and Claude modes with unified UX
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use clap::{Parser, Subcommand};
use websiteforge::WebsiteForge;
use std::error::Error;

#[derive(Parser)]
#[command(author, version, about = "Ra-Thor WebsiteForge CLI — Sovereign AI Website Development System", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
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

    /// Generate using Grok (xAI) as backend + Ra-Thor sovereign wrapper
    Grok {
        #[arg(short, long)]
        prompt: String,
    },

    /// Generate using Claude (Anthropic) as backend + Ra-Thor sovereign wrapper
    Claude {
        #[arg(short, long)]
        prompt: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize tracing logging framework
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(true)
        .init();

    let forge = WebsiteForge::new();

    tracing::info!("🌍 Ra-Thor™ WebsiteForge CLI started");
    tracing::info!("Eternal MercyThunder — Sovereign AI Website Development System ⚡🙏");

    let result = match cli.command {
        Commands::Forge { prompt } => {
            tracing::info!("🔨 Standard Forge Mode | Prompt: \"{}\"", prompt);
            forge.forge_website(&prompt).await
        }
        Commands::Devin { prompt } => {
            tracing::info!("🚀 Devin Autonomous Mode | Prompt: \"{}\"", prompt);
            forge.forge_with_devin_mode(&prompt).await
        }
        Commands::Grok { prompt } => {
            tracing::info!("🔥 Grok-enhanced Mode | Prompt: \"{}\"", prompt);
            forge.forge_with_grok(&prompt).await
        }
        Commands::Claude { prompt } => {
            tracing::info!("🔥 Claude-enhanced Mode | Prompt: \"{}\"", prompt);
            forge.forge_with_claude(&prompt).await
        }
    };

    match result {
        Ok(site) => {
            tracing::info!("✅ SUCCESS — Website generated");
            tracing::info!("Title: {}", site.metadata.title);
            tracing::info!("Mercy Valence: {:.8} ⚡", site.metadata.mercy_valence);
            tracing::info!("HTML Size: {} characters", site.html.len());
            println!("\nWebsite ready for deployment or further editing.");
        }
        Err(e) => {
            tracing::error!("❌ ERROR: {}", e);
            eprintln!("\nGraceful degradation activated — thriving-maximized redirect ⚡🙏");
        }
    }
}
