// crates/xtask/src/main.rs
// Ra-Thor™ xtask — Sovereign Monorepo Automation Hub

use clap::{Parser, Subcommand};
use ra_thor_mercy::MercyEngine;
use std::process::{self, Command};
use thiserror::Error;
use tracing::error;

#[derive(Error, Debug)]
pub enum XtaskError {
    #[error("Cargo command '{command}' failed")]
    Cargo { command: String, #[source] source: std::io::Error },
    #[error("MercyEngine error during {context}")]
    Mercy { context: String, #[source] source: ra_thor_mercy::MercyError },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Command failed with exit code {0}")]
    CommandFailed(i32),
    #[error("Validation failed: {reason}")]
    Validation { reason: String },
}

type Result<T> = std::result::Result<T, XtaskError>;

struct MercyGuard<'a> { engine: &'a MercyEngine }

impl<'a> MercyGuard<'a> {
    fn new(engine: &'a MercyEngine) -> Self { Self { engine } }
    async fn run<F, Fut, T>(&self, context: &str, operation: F) -> Result<T>
    where F: FnOnce() -> Fut, Fut: std::future::Future<Output = Result<T>> {
        tracing::info!("🛡️ MercyGuard: {}", context);
        let result = operation().await;
        result
    }
}

#[derive(Parser)]
#[command(author, version, about = "Ra-Thor Sovereign Monorepo Automation Hub")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    #[arg(short, long, global = true)] verbose: bool,
    #[arg(short, long, global = true)] dry_run: bool,
}

#[derive(Subcommand)]
enum Commands {
    Upgrade, Reorganize, MercyCheck,
    Commit { message: String },
    Merge { base: String, ours: String, theirs: String },
    Format, Lint, Test, Build, Forge { prompt: String },
    FullSync, Deploy { dry_run: bool }, UpgradeDeps,
    ForgeDeploy { prompt: String, platform: Option<String> },
    Clean, Doc, Validate, Audit, Outdated, Status,

    /// Build a Sovereign Shard
    BuildShard { profile: String, release: bool },

    /// List available shard profiles
    ListShards,

    /// Check a shard profile (cargo check)
    CheckShard { profile: String },

    /// Test a shard profile
    TestShard { profile: String },
}

fn run_cargo_command(args: &[&str], description: &str) -> Result<()> {
    println!("🔧 cargo {}", args.join(" "));
    let status = Command::new("cargo").args(args).status()
        .map_err(|e| XtaskError::Cargo { command: args.join(" "), source: e })?;
    if status.success() { println!("✅ {}", description); Ok(()) }
    else { Err(XtaskError::CommandFailed(status.code().unwrap_or(-1))) }
}

fn get_feature(profile: &str) -> String {
    match profile {
        "full" => "full".to_string(),
        "focused-real-estate" | "real-estate" => "focused-real-estate".to_string(),
        "focused-geometry" | "geometry" => "focused-geometry".to_string(),
        _ => { eprintln!("Unknown profile '{}', defaulting to full", profile); "full".to_string() }
    }
}

fn build_shard(profile: &str, release: bool) -> Result<()> {
    let feature = get_feature(profile);
    let mut args = vec!["build", "-p", "shard-composer", "--features", &feature];
    if release { args.push("--release"); }
    run_cargo_command(&args, &format!("Building shard '{}'", profile))
}

fn check_shard(profile: &str) -> Result<()> {
    let feature = get_feature(profile);
    run_cargo_command(&["check", "-p", "shard-composer", "--features", &feature], &format!("Checking shard '{}'", profile))
}

fn test_shard(profile: &str) -> Result<()> {
    let feature = get_feature(profile);
    run_cargo_command(&["test", "-p", "shard-composer", "--features", &feature], &format!("Testing shard '{}'", profile))
}

fn list_shards() {
    println!("Available Sovereign Shard profiles:");
    println!("  full                  → Complete ONE Organism");
    println!("  focused-real-estate   → Real Estate + Professional Judgment");
    println!("  focused-geometry      → Geometry + Riemannian");
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    let cli = Cli::parse();
    let engine = MercyEngine::new();
    let guard = MercyGuard::new(&engine);

    let result = match cli.command {
        Commands::BuildShard { profile, release } => build_shard(&profile, release),
        Commands::CheckShard { profile } => check_shard(&profile),
        Commands::TestShard { profile } => test_shard(&profile),
        Commands::ListShards => { list_shards(); Ok(()) }
        _ => { println!("Command not fully implemented in this view"); Ok(()) }
    };

    match result {
        Ok(_) => process::exit(0),
        Err(e) => { error!("xtask error: {}", e); process::exit(1); }
    }
}
