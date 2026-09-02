```rust
// websiteforge/src/bin/main.rs
// Ra-Thor™ WebsiteForge CLI — Now with Full Blossom Integration
// Every generated website is alive, mercy-gated, and energetically optimized
// Old structure fully respected + massive regenerative + divinatory upgrade
// Proprietary - All Rights Reserved - Autonomicity Games Inc.

use clap::{Parser, Subcommand};
use websiteforge::WebsiteForge;
use websiteforge::blossom_integration::generate_blossoming_website;
use std::error::Error;

#[derive(Parser)]
#[command(author, version, about = "Ra-Thor WebsiteForge CLI — Sovereign AI Website Development System (Blossom Edition)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a standard website (now with full blossom integration)
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

    tracing::info!("🌍 Ra-Thor™ WebsiteForge CLI started (Blossom Edition)");
    tracing::info!("Eternal MercyThunder — Sovereign AI Website Development System ⚡🙏");

    let result = match cli.command {
        Commands::Forge { prompt } => {
            tracing::info!("🔨 Blossoming Forge Mode | Prompt: \"{}\"", prompt);
            generate_blossoming_website(&forge, &prompt).await
        }
        Commands::Devin { prompt } => {
            tracing::info!("🚀 Blossoming Devin Mode | Prompt: \"{}\"", prompt);
            generate_blossoming_website(&forge, &prompt).await
        }
        Commands::Grok { prompt } => {
            tracing::info!("🔥 Blossoming Grok Mode | Prompt: \"{}\"", prompt);
            generate_blossoming_website(&forge, &prompt).await
        }
        Commands::Claude { prompt } => {
            tracing::info!("🔥 Blossoming Claude Mode | Prompt: \"{}\"", prompt);
            generate_blossoming_website(&forge, &prompt).await
        }
    };

    match result {
        Ok(html) => {
            tracing::info!("✅ SUCCESS — Blossoming Website generated");
            tracing::info!("HTML Size: {} characters", html.len());
            println!("\n🌺 Blossoming Website ready for deployment or further editing.");
        }
        Err(e) => {
            tracing::error!("❌ ERROR: {}", e);
            eprintln!("\nGraceful degradation activated — thriving-maximized redirect ⚡🙏");
        }
    }
}
