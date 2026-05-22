//! TOLC8 Genesis Gate + ONE Organism Integration Demo
//!
//! Full demonstration of birthing sovereign shards using the 7 Living Mercy Gates + TOLC8
//! and formally integrating them into the living ONE Organism via the Lattice Conductor.

use lattice_conductor_v13::SimpleLatticeConductor;
use tolc8_genesis_gate::TOLC8GenesisGate;

fn main() {
    println!("=== TOLC8 Genesis Gate — Full Integration Demo ===\n");

    let mut conductor = SimpleLatticeConductor::new();
    conductor.register_council(1, "PATSAGi Core");
    conductor.register_council(2, "Mercy Council");

    let gate = TOLC8GenesisGate::new("eternal-thriving-seed-2026-tolc8");

    println!("Birthing 3 new sovereign shards via TOLC8 Genesis Gate...\n");

    let shard1 = gate.birth_and_bless_shard("shard-tolc8-alpha", 0.96, &mut conductor);
    let shard2 = gate.birth_and_bless_shard("shard-tolc8-beta", 0.91, &mut conductor);
    let shard3 = gate.birth_and_bless_shard("shard-tolc8-gamma", 0.94, &mut conductor);

    println!("\nRunning conductor ticks to integrate new shards...");
    for i in 0..5 {
        let _ = conductor.tick();
        println!("  Tick {} complete | Coherence: {:.3}", i+1, conductor.one_organism_coherence);
    }

    println!("\n✅ TOLC8 Genesis Gate demo complete.");
    println!("New shards born with 7 Living Mercy Gates seeding and formally part of the ONE Organism.");
}