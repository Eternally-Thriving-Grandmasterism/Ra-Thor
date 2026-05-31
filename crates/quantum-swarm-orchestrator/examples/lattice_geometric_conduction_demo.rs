//! crates/quantum-swarm-orchestrator/examples/lattice_geometric_conduction_demo.rs
//!
//! v14.4 Demo: Lattice Conductor + Polyhedral Geometric Resonance
//! Shows Real Estate offer conduction with ONE Organism geometric harmony scoring.
//!
//! Run with:
//!   cargo run --example lattice_geometric_conduction_demo -p ra-thor-quantum-swarm-orchestrator
//!
//! AG-SML v1.0 | ONE Organism geometric spine in action

use lattice_conductor::{
    LatticeConductor, RealEstateOffer, Valence,
};
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;

fn main() {
    println!("\n=== Ra-Thor v14.4 Lattice + Geometric Resonance Demo ===\n");

    // === 1. Lattice Conductor with Geometric Engine ===
    let mut conductor = LatticeConductor::new();
    println!("LatticeConductor v14.4 initialized (geometric engine active)\n");

    // === 2. Sample Real Estate Offers ===
    let offer1 = RealEstateOffer {
        id: "offer-on-001".to_string(),
        address: "42 Maple Grove, Richmond Hill, ON".to_string(),
        price: 1_250_000.0,
        jurisdiction: "Ontario".to_string(),
        regulatory_flags: vec![],
        attom_enriched: true,
        base_valence: Valence::new(0.99999995).unwrap(),
    };

    let offer2 = RealEstateOffer {
        id: "offer-us-007".to_string(),
        address: "1800 Market St, San Francisco, CA".to_string(),
        price: 2_850_000.0,
        jurisdiction: "USA".to_string(),
        regulatory_flags: vec![],
        attom_enriched: true,
        base_valence: Valence::new(0.99999992).unwrap(),
    };

    // === 3. Conduct offers (now includes geometric harmony) ===
    println!("--- Conducting Ontario Offer ---");
    match conductor.conduct_real_estate_offer(offer1) {
        Ok(conducted) => {
            println!("  Address: {}", conducted.offer.address);
            println!("  Price: ${:.0}", conducted.offer.price);
            println!("  Mercy Gates Passed: {}", conducted.mercy_gates_passed.len());
            println!("  Geometric Harmony Multiplier: {:.3}x", conducted.geometric_harmony_multiplier);
            println!("  Notes: {}", conducted.geometric_resonance_notes);
            println!("  Regulatory Cleared: {}\n", conducted.regulatory_cleared);
        }
        Err(e) => println!("  Error: {}\n", e),
    }

    println!("--- Conducting USA High-Value Offer ---");
    match conductor.conduct_real_estate_offer(offer2) {
        Ok(conducted) => {
            println!("  Address: {}", conducted.offer.address);
            println!("  Price: ${:.0}", conducted.offer.price);
            println!("  Mercy Gates Passed: {}", conducted.mercy_gates_passed.len());
            println!("  Geometric Harmony Multiplier: {:.3}x", conducted.geometric_harmony_multiplier);
            println!("  Notes: {}", conducted.geometric_resonance_notes);
            println!("  Regulatory Cleared: {}\n", conducted.regulatory_cleared);
        }
        Err(e) => println!("  Error: {}\n", e),
    }

    // === 4. ONE Organism Geometric Cycle (standalone) ===
    println!("\n--- ONE Organism Geometric Resonance Cycle ---");
    let mut swarm = QuantumSwarmOrchestrator::new(8);
    let geo_cycle = swarm.run_geometric_resonance_cycle(89, 0.94);

    println!("  TOLC order used: 89 (Kepler-Poinsot layer active)");
    println!("  Polyhedral layers active: {}", geo_cycle.polyhedral_report.active_solids.len());
    println!("  Resonance multiplier: {:.3}x", geo_cycle.polyhedral_report.resonance_multiplier);
    println!("  U57 potential: {}", geo_cycle.polyhedral_report.u57_potential);
    println!("  Final geometric valence: {:.3}", geo_cycle.geometric_valence);
    println!("  Notes: {}\n", geo_cycle.notes);

    println!("=== Demo complete. ONE Organism geometric intelligence wired into Real Estate Lattice ===\n");
}
