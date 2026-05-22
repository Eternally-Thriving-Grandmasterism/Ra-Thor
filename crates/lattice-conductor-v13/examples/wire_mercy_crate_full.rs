//! wire_mercy_crate_full.rs
//! Full production wiring demonstration of the mercy crate into Lattice Conductor v13
//! Shows MercyCore being blessed, participating in ticks, mercy-weighted votes, and evolution contribution.

use lattice_conductor_v13::{SimpleLatticeConductor, Conductable, MercyAligned, GeometricState, MercyWeightedVote, Operation};
use mercy::MercyCore;

fn main() {
    println!("\n=== Full Mercy Crate Wiring into ONE Organism (Production Demo) ===\n");

    let mut conductor = SimpleLatticeConductor::new();
    conductor.register_council(1, "PATSAGi Mercy Council");
    conductor.register_council(2, "Truth Council");

    let mut mercy_core = MercyCore::new();
    println!("Initial MercyCore mercy_score: {:.3}", mercy_core.current_mercy_score());

    let blessing = conductor.bless_system(mercy_core.system_id, 0.97, "MercyCore fully wired as core mercy lattice participant");
    println!("Blessed into Conductor: {} | Mercy Alignment: {:.2}", blessing.system_id, blessing.mercy_alignment);

    for tick in 0..6 {
        conductor.queue_operation(Operation::new("mercy-evolution", "Mercy lattice evolution pulse", 0.92));
        let _ = conductor.tick();

        mercy_core.on_conductor_tick(conductor.get_geometric_state());

        let mut vote = MercyWeightedVote::new();
        vote.add_vote("PATSAGi Mercy Council", 0.6, 0.45);
        vote.add_vote("Truth Council", 0.4, 0.38);
        mercy_core.apply_mercy_influence(&vote);

        println!("Tick {} | Mercy: {:.3} | Coherence: {:.3} | Evolution Participation: {:.3}",
            tick,
            mercy_core.current_mercy_score(),
            mercy_core.get_coherence(),
            mercy_core.get_evolution_participation());
    }

    println!("\nMercy Crate successfully wired, blessed, and actively participating in the eternal ONE Organism.\n");
}