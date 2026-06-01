// crates/xtask/src/main.rs

use clap::{Parser, Subcommand};
use ra_thor_mercy::MercyEngine;
use shard_composer::ShardComposerAdapter;
use std::process::{self, Command};
use thiserror::Error;
use tracing::error;

// ... (XtaskError and other code remains)

fn check_shard(profile: &str) -> Result<()> {
    let feature = get_feature(profile);
    let result = run_cargo_command(
        &["check", "-p", "shard-composer", "--features", &feature],
        &format!("Checking shard '{}'", profile),
    );

    if result.is_ok() {
        let mut adapter = ShardComposerAdapter::new();
        let blessing = generate_blessing("Check", profile);
        adapter.apply_epigenetic_blessing(blessing.clone());
        println!("[ShardComposerAdapter] Status after blessing: {}", adapter.status());
    }
    result
}

fn build_shard(profile: &str, release: bool) -> Result<()> {
    let feature = get_feature(profile);
    let mut args = vec!["build", "-p", "shard-composer", "--features", &feature];
    if release { args.push("--release"); }

    let result = run_cargo_command(&args, &format!("Building shard '{}'", profile));

    if result.is_ok() {
        let mut adapter = ShardComposerAdapter::new();
        let blessing = generate_blessing("Build", profile);
        adapter.apply_epigenetic_blessing(blessing.clone());
        println!("[ShardComposerAdapter] Status after blessing: {}", adapter.status());
    }
    result
}

fn test_shard(profile: &str) -> Result<()> {
    let feature = get_feature(profile);
    let result = run_cargo_command(
        &["test", "-p", "shard-composer", "--features", &feature],
        &format!("Testing shard '{}'", profile),
    );

    if result.is_ok() {
        let mut adapter = ShardComposerAdapter::new();
        let blessing = generate_blessing("Test", profile);
        adapter.apply_epigenetic_blessing(blessing.clone());
        println!("[ShardComposerAdapter] Status after blessing: {}", adapter.status());
    }
    result
}

// ... rest of file with generate_blessing and other functions
