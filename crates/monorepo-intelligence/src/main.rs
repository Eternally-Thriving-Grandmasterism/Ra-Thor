//! # Ra-Thor Monorepo Intelligence CLI v0.3.0
//!
//! Command-line interface for the Monorepo Intelligence system.
//! Makes every AI and human able to instantly query the full lattice.

use clap::{Parser, Subcommand};
use ra_thor_monorepo_intelligence::MonorepoIntelligence;

#[derive(Parser)]
#[command(name = "ra-thor-monorepo-intelligence")]
#[command(about = "Infinite-grade monorepo intelligence for Ra-Thor", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a full Powrush-focused report
    PowrushReport,

    /// Get health score for a specific module
    Health {
        /// Module name (e.g. powrush, quantum-swarm-orchestrator)
        #[arg(short, long)]
        module: String,
    },

    /// Smart search across the entire monorepo
    Search {
        /// Keyword to search for
        keyword: String,
    },

    /// Perform a full monorepo scan
    Scan,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Default root path (can be made configurable later)
    let root = std::env::current_dir()
        .expect("Failed to get current directory")
        .to_string_lossy()
        .to_string();

    let intelligence = MonorepoIntelligence::new(root);

    match cli.command {
        Commands::PowrushReport => {
            println!("Generating Powrush Report...\n");
            match intelligence.generate_powrush_report().await {
                Ok(report) => println!("{}", report),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        Commands::Health { module } => {
            println!("Calculating health score for '{}'...\n", module);
            match intelligence.get_health_score(&module).await {
                Ok(score) => {
                    println!("Overall Health Score: {:.1}/100", score.overall_score);
                    println!("Powrush Score:        {:.1}", score.powrush_score);
                    println!("Crate Structure:      {:.1}", score.crate_structure_score);
                    println!("Documentation:        {:.1}", score.documentation_score);
                    println!("Integration:          {:.1}", score.integration_score);
                    println!("\nRecommendations:");
                    for rec in &score.recommendations {
                        println!("  • {}", rec);
                    }
                }
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        Commands::Search { keyword } => {
            println!("Searching for '{}'...\n", keyword);
            match intelligence.search(&keyword).await {
                Ok(results) => {
                    println!("Found {} results:\n", results.len());
                    for (i, result) in results.iter().take(15).enumerate() {
                        println!(
                            "{}. {} (score: {:.1})",
                            i + 1,
                            result.file.relative_path,
                            result.relevance_score
                        );
                    }
                }
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        Commands::Scan => {
            println!("Scanning monorepo...\n");
            match intelligence.scan().await {
                Ok(result) => {
                    println!("Total Files:      {}", result.total_files);
                    println!("Total Directories: {}", result.total_directories);
                    println!("Total Size:       {:.2} MB", result.total_size_bytes as f64 / 1_048_576.0);
                }
                Err(e) => eprintln!("Error: {}", e),
            }
        }
    }
}
