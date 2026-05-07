//! Council Simulator Binary
//!
//! A simple command-line tool to run PATSAGi Council sessions.
//! This is the entry point to experience the 13+ Councils in action.

use council::{
    CouncilProposal, CouncilSession, CouncilSessionResult,
    deliberation::run_parallel_deliberation,
    voting::conduct_voting,
    coherence::compute_session_coherence,
    outcome_applicator::apply_outcome_to_lattice,
};

use patsagi_councils::CouncilMember;
use ra_thor_mercy::MercyGateEvaluator;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmBridge;
use ra_thor_kernel::Kernel;

use uuid::Uuid;
use chrono::Utc;

#[tokio::main]
async fn main() {
    println!("══════════════════════════════════════════════════════════════");
    println!("           PATSAGi Council Simulator v0.3.7");
    println!("           13+ Parallel Living Architectural Designers");
    println!("══════════════════════════════════════════════════════════════\n");

    // === Demo Proposal ===
    let proposal = CouncilProposal {
        id: Uuid::new_v4(),
        title: "Increase resource allocation to interstellar propulsion research by 18%".to_string(),
        description: "Proposal to boost interstellar propulsion R&D while maintaining strong mercy alignment with current Powrush economy and TOLC principles.".to_string(),
        complexity: 0.78,
        impact_level: 0.85,
    };

    println!("📜 Proposal Received:");
    println!("   Title: {}", proposal.title);
    println!("   Description: {}", proposal.description);
    println!("   Complexity: {:.2} | Impact: {:.2}\n", proposal.complexity, proposal.impact_level);

    // === Initialize Council Session ===
    // In a real run, these would be loaded from persistent state or configuration
    let members = load_demo_council_members();
    let mercy_evaluator = MercyGateEvaluator::default();
    let quantum_swarm_bridge = QuantumSwarmBridge::new();
    let mut kernel = Kernel::new();

    let mut session = CouncilSession::new(
        members,
        mercy_evaluator,
        quantum_swarm_bridge,
        kernel,
    );

    // === Run the Council Session ===
    println!("🧠 Council deliberation in progress...\n");
    let result = session.run_session(proposal.clone()).await;

    // === Display Results ===
    print_session_result(&result);

    println!("\n✅ Council session completed.");
    println!("══════════════════════════════════════════════════════════════");
}

/// Loads a small set of demo council members for simulation purposes.
fn load_demo_council_members() -> Vec<patsagi_councils::CouncilMember> {
    vec![
        patsagi_councils::CouncilMember::new(1, "Harmony Weaver", "Ethics & Thriving"),
        patsagi_councils::CouncilMember::new(2, "Truth Seeker", "Pure Truth & Clarity"),
        patsagi_councils::CouncilMember::new(3, "Abundance Keeper", "Resource Flow & Sustainability"),
        patsagi_councils::CouncilMember::new(4, "Sovereign Guardian", "Autonomy & Radical Love"),
        patsagi_councils::CouncilMember::new(5, "TOLC Anchor", "TOLC Resonance & Mathematical Harmony"),
    ]
}

fn print_session_result(result: &CouncilSessionResult) {
    println!("📊 Council Session Result");
    println!("──────────────────────────────────────────────────────────────");
    println!("Session ID      : {}", result.session_id);
    println!("Proposal        : {}", result.proposal.title);
    println!("Passed          : {}", if result.passed { "✅ YES" } else { "❌ NO" });
    println!("Final Coherence : {:.4}", result.final_coherence);
    println!("Mercy Valence   : {:.4}", result.mercy_valence);
    println!("Timestamp       : {}", result.timestamp);
    println!("──────────────────────────────────────────────────────────────");
}
