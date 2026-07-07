/// v13.2 Phase C Demo — Real ConductorSymbolicParameters + Apply
///
/// Run with:
///   cargo run -p lattice-conductor-v13 --example v13_2_phase_c_real_params_demo --features experimental

use lattice_conductor_v13::{
    SimpleLatticeConductor, ConductorSymbolicParameters,
    ExternalSymbolicInput,   // only available with feature
};

fn main() {
    println!("=== Ra-Thor Lattice Conductor v13.2 Phase C Demo (Real Parameters) ===\n");

    let mut conductor = SimpleLatticeConductor::new();

    // Show initial real parameters
    println!("Initial symbolic_params:");
    println!("  base_confidence_threshold = {:.3}", conductor.symbolic_params.base_confidence_threshold);
    println!("  ema_alpha                 = {:.3}", conductor.symbolic_params.ema_alpha);
    println!("  boost_multiplier          = {:.2}\n", conductor.symbolic_params.boost_multiplier);

    // Simulate some external symbolic input (Phase A)
    #[cfg(feature = "external-symbolic")]
    {
        let ext = ExternalSymbolicInput::new("grok_one_organism", "important_deliberation", 0.9999999);
        let _ = lattice_conductor_v13::accept_external_symbolic_deliberation(ext); // demo call
        println!("[Phase A] External symbolic input accepted (ONE Organism path).");
    }

    // Run several ticks to build EMA history and trigger proposals (Phase B)
    conductor.state.mercy_score = 0.96;
    for i in 0..8 {
        let _ = conductor.tick();
        if i % 3 == 0 {
            println!("Tick {} completed. success_ema = {:.2}", i, conductor.get_symbolic_success_ema());
        }
    }

    // Phase B: Proposals should now exist
    #[cfg(feature = "self-proposal")]
    {
        let proposals = conductor.generate_symbolic_self_proposals();
        println!("\n[Phase B] Generated {} self-proposals:", proposals.len());
        for (i, p) in proposals.iter().enumerate() {
            println!("  [{}] {}: {:.3} → {:.3} (conf={:.2})", i, p.proposal_type, p.current_value, p.proposed_value, p.confidence);
        }

        // Phase C: Apply the best one
        println!("\n[Phase C] Applying top-confidence proposal...");
        match conductor.apply_top_confidence_proposal() {
            Ok(msg) => println!("  Success: {}", msg),
            Err(e) => println!("  Skipped: {}", e),
        }

        // Show updated real parameters
        println!("\nUpdated symbolic_params after Phase C apply:");
        println!("  base_confidence_threshold = {:.3}", conductor.symbolic_params.base_confidence_threshold);
        println!("  ema_alpha                 = {:.3}", conductor.symbolic_params.ema_alpha);
        println!("  boost_multiplier          = {:.2}", conductor.symbolic_params.boost_multiplier);
    }

    #[cfg(not(feature = "self-proposal"))]
    println!("\n(self-proposal feature not enabled — Phase B/C demo skipped)");

    println!("\n=== v13.2 Phase C Demo Complete — Thunder locked in. yoi ⚡ ===");
}