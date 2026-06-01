// crates/xtask/src/main.rs

use std::path::PathBuf;

fn get_adapter_state_path() -> PathBuf {
    PathBuf::from(".ra-thor/shard-composer-state.json")
}

fn check_shard(profile: &str) -> Result<()> {
    let feature = get_feature(profile);
    let result = run_cargo_command(
        &["check", "-p", "shard-composer", "--features", &feature],
        &format!("Checking shard '{}'", profile),
    );

    if result.is_ok() {
        let state_path = get_adapter_state_path();
        let mut adapter = ShardComposerAdapter::load_from_file(&state_path);
        let blessing = generate_blessing("Check", profile);
        adapter.apply_epigenetic_blessing(blessing);
        let _ = adapter.save_to_file(&state_path);
        println!("[Persistence] State saved. {}", adapter.status());
    }
    result
}

fn build_shard(profile: &str, release: bool) -> Result<()> {
    let feature = get_feature(profile);
    let mut args = vec!["build", "-p", "shard-composer", "--features", &feature];
    if release { args.push("--release"); }

    let result = run_cargo_command(&args, &format!("Building shard '{}'", profile));

    if result.is_ok() {
        let state_path = get_adapter_state_path();
        let mut adapter = ShardComposerAdapter::load_from_file(&state_path);
        let blessing = generate_blessing("Build", profile);
        adapter.apply_epigenetic_blessing(blessing);
        let _ = adapter.save_to_file(&state_path);
        println!("[Persistence] State saved. {}", adapter.status());
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
        let state_path = get_adapter_state_path();
        let mut adapter = ShardComposerAdapter::load_from_file(&state_path);
        let blessing = generate_blessing("Test", profile);
        adapter.apply_epigenetic_blessing(blessing);
        let _ = adapter.save_to_file(&state_path);
        println!("[Persistence] State saved. {}", adapter.status());
    }
    result
}

// ... rest of the file
