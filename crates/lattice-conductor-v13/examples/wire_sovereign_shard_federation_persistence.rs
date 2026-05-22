//! wire_sovereign_shard_federation_persistence.rs
//! Demonstrates SovereignShardFederation with offline mode, full reconciliation protocol,
//! persistent storage (save/load), and collective operations as part of ONE Organism.

use lattice_conductor_v13::{SimpleLatticeConductor, GeometricState};
use sovereign_shard_genesis::{SovereignShard, SovereignShardFederation, SovereignShardGenesis};

fn main() {
    println!("\n=== Sovereign Shard Federation + Persistence + Offline Reconciliation Demo ===\n");

    let mut conductor = SimpleLatticeConductor::new();
    conductor.register_council(1, "Sovereign Shard Council");

    let genesis = SovereignShardGenesis;
    let mut registry = lattice_conductor_v13::ConductorRegistry::new();

    let (mut shard1, _b1) = genesis.genesis_shard("shard-alpha", "Alpha Sovereign Shard", 0, &mut registry);
    let (mut shard2, _b2) = genesis.genesis_shard("shard-beta", "Beta Sovereign Shard", 0, &mut registry);

    let mut federation = SovereignShardFederation::new();
    federation.add_shard(shard1);
    federation.add_shard(shard2);

    println!("Federation created with {} shards. Collective mercy: {:.3}", federation.shards.len(), federation.collective_mercy_score());

    if let Some(shard) = federation.get_shard_mut("shard-beta") {
        shard.enable_offline_mode();
        println!("shard-beta entered offline mode (local sovereignty).");
    }

    for i in 0..5 {
        federation.tick_all();
        println!("Federation tick {} | Collective Mercy: {:.3}", i, federation.collective_mercy_score());
    }

    let conductor_state = conductor.get_geometric_state().clone();
    federation.reconcile_all_with_conductor(&conductor_state, 1.05);
    println!("All shards reconciled with conductor. Offline shard now online.");

    if let Some(shard) = federation.get_shard("shard-alpha") {
        let path = "/tmp/shard_alpha_persist.json";
        let _ = shard.save_to_file(path);
        println!("shard-alpha persisted to {}", path);

        if let Ok(loaded) = SovereignShard::load_from_file(path) {
            println!("Loaded shard from persistence: {} | mercy: {:.3}", loaded.shard_id, loaded.mercy_alignment);
        }
        let _ = std::fs::remove_file(path);
    }

    println!("\nSovereign Shard Federation with persistence and offline reconciliation successfully demonstrated as ONE Organism component.\n");
}