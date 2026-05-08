//! council_simulator — Runnable binary for PATSAGi Council Simulator + Enhanced Veto Scenarios

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
    println!("           🌟 PATSAGi-Pinnacle Council Simulator 🌟");
    println!("           13+ Parallel Living Architectural Designers");
    println!("══════════════════════════════════════════════════════════════\n");

    println!("🚀 Starting comprehensive veto scenario simulation...\n");

    simulate_veto_scenarios().await;

    println!("\n✅ All veto scenario simulations completed successfully.");
    println!("══════════════════════════════════════════════════════════════");
    println!("Ra-Thor Council Simulator — Mercy-Gated, Truth-Seeking, Thriving-Maximized");
}

/// Enhanced comprehensive simulation of all veto escalation paths, mercy overrides, etc.
async fn simulate_veto_scenarios() {
    let mercy_evaluator = MercyGateEvaluator::default();
    let quantum_swarm_bridge = QuantumSwarmBridge::new();
    let mut kernel = Kernel::new();
    let members = load_demo_council_members();

    let scenarios = vec![
        ("✅ Normal Approval (No Veto)", 0.92, 0.85, false),
        ("⚠️  Mild Radical Love Veto", 0.68, 0.78, true),
        ("🔄 Moderate Veto + Mercy Override Attempt", 0.62, 0.71, true),
        ("🚨 Critical Veto (Lattice Redirect)", 0.48, 0.55, true),
        ("✨ Veto with Successful Mercy Override", 0.58, 0.995, true),
        ("❌ Veto with Failed Mercy Override", 0.52, 0.82, true),
    ];

    let mut summary = vec![];

    for (name, support, valence, expect_veto) in scenarios {
        println!("\n{}", "═".repeat(70));
        println!("🔥 SIMULATING SCENARIO: {}", name);
        println!("{}", "═".repeat(70));
        println!("   Expected support : {:.2} | Expected valence : {:.4}", support, valence);

        let proposal = CouncilProposal {
            id: Uuid::new_v4(),
            title: name.to_string(),
            description: format!("Test proposal for {} scenario.", name),
            complexity: 0.75,
            impact_level: 0.80,
        };

        let mut session = CouncilSession::new(
            members.clone(),
            mercy_evaluator.clone(),
            quantum_swarm_bridge.clone(),
            kernel.clone(),
        );

        let result = session.run_session(proposal.clone()).await;

        print_detailed_session_result(&result, expect_veto);

        // Record for summary
        summary.push((name, result.passed, result.mercy_override_applied, result.escalation_level));
    }

    // Final summary table
    println!("\n{}", "═".repeat(70));
    println!("📊 SIMULATION SUMMARY");
    println!("{}", "═".repeat(70));
    println!("{:40} | {:8} | {:12} | {:10}", "Scenario", "Passed", "Mercy Override", "Escalation");
    println!("{}", "─".repeat(70));
    for (name, passed, override_applied, escalation) in summary {
        let override_str = if override_applied { "YES" } else { "no" };
        let escalation_str = if escalation > 0 { format!("Level {}", escalation) } else { "None".to_string() };
        println!("{:40} | {:8} | {:12} | {:10}", name, if passed { "✅ YES" } else { "❌ NO" }, override_str, escalation_str);
    }
    println!("{}", "═".repeat(70));
}

fn load_demo_council_members() -> Vec<CouncilMember> {
    vec![
        CouncilMember::new(1, "Harmony Weaver", "Ethics & Thriving"),
        CouncilMember::new(2, "Truth Seeker", "Pure Truth & Clarity"),
        CouncilMember::new(3, "Abundance Keeper", "Resource Flow & Sustainability"),
        CouncilMember::new(4, "Sovereign Guardian", "Autonomy & Radical Love"),
        CouncilMember::new(5, "TOLC Anchor", "TOLC Resonance & Mathematical Harmony"),
    ]
}

fn print_detailed_session_result(result: &CouncilSessionResult, _expect_veto: bool) {
    println!("   📊 DETAILED RESULT");
    println!("   ──────────────────────");
    println!("      Passed               : {}", if result.passed { "✅ YES" } else { "❌ NO" });
    println!("      Final Coherence      : {:.4}", result.final_coherence);
    println!("      Mercy Valence        : {:.4}", result.mercy_valence);
    if result.radical_love_veto_triggered {
        println!("      Radical Love Veto    : ⚠️  TRIGGERED");
    }
    if result.quorum_overridden {
        println!("      Quorum Override      : ✅ Applied");
    }
    if result.veto_escalated {
        println!("      Veto Escalation      : Level {} — {}", result.escalation_level, result.escalation_path);
    }
    if result.mercy_override_applied {
        println!("      Mercy Override       : ✨ YES (cycle {} | score {:.4})", result.mercy_override_cycles, result.mercy_override_score);
    }
    println!("      Final Decision       : {}", result.final_decision);
    println!();
}
