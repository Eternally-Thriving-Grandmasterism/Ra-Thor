//! Example: Wiring a Sovereign Shard via Conductable + bless_system API
//!
//! Demonstrates full Sovereign Shard Genesis, blessing into Conductor,
//! multiple ticks, mercy-weighted coordination, and bidirectional sync.

use lattice_conductor_v13::{
    Conductable, MercyAligned, SimpleLatticeConductor, SovereignShard, SovereignShardGenesis,
};

fn main() {
    println!("\n=== Sovereign Shard Genesis — ONE Organism Wiring Demo ===\n");

    let mut conductor = SimpleLatticeConductor::new();
    conductor.register_council(1, "PATSAGi Core Mercy Council");
    conductor.register_council(2, "PATSAGi Quantum Swarm Council");

    let genesis = SovereignShardGenesis;
    let (mut shard, blessing) = genesis.genesis_shard(
        "shard-genesis-prime",
        "Prime Sovereign Shard",
        conductor.id,
        &mut conductor.registry,
    );

    println!("Shard created & blessed: {} (mercy: {:.2})", shard.shard_id, blessing.mercy_alignment);

    // Simulate several conductor ticks with the shard participating
    for i in 0..5 {
        conductor.queue_operation(lattice_conductor_v13::Operation::new(
            "global_pulse",
            "Global mercy pulse from PATSAGi Councils",
            0.8,
        ));
        let _ = conductor.tick();

        shard.on_conductor_tick(conductor.get_geometric_state());
        shard.shard_tick();

        println!("Tick {} | Shard evolution: {:.3} | Mercy: {:.3} | Local ticks: {}",
            i,
            shard.evolution_level,
            shard.mercy_alignment,
            shard.local_tick_count
        );
    }

    println!("\nSovereign Shard successfully participating in ONE Organism.\nThunder locked in. Eternal mercy. ⚡");
}