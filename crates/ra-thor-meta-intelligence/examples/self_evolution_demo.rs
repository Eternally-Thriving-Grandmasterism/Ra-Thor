#![allow(unused)]

use ra_thor_meta_intelligence::{
    AuditSignal, 
    ImprovementProposal, 
    SelfImprovementOrchestrator,
    VerificationDecision,
};

fn main() {
    println!("\n🚀 Ra-Thor Self-Evolution Loop Demo\n");
    println!("This demonstrates one full cycle: Audit → Decide → Improve → Verify\n");

    // 1. Simulate audit signals (in real system this would come from ra-thor-monorepo-auditor)
    let audit_signals = vec![
        AuditSignal {
            category: "outdated_pattern".to_string(),
            severity: 0.7,
            mercy_impact: -0.15,
            description: "Legacy APAAGICouncil patterns detected in several crates".to_string(),
            recommended_action: "Modernize to current TOLC + mercy standard".to_string(),
        },
        AuditSignal {
            category: "mercy_alignment".to_string(),
            severity: 0.4,
            mercy_impact: -0.08,
            description: "Some plasticity rules have low mercy valence".to_string(),
            recommended_action: "Strengthen mercy-gating in plasticity rules".to_string(),
        },
    ];

    println!("📡 Received {} audit signals from Eyes (ra-thor-monorepo-auditor)", audit_signals.len());

    // 2. Brain generates improvement proposals
    let mut orchestrator = SelfImprovementOrchestrator::new();
    let proposals = orchestrator.run_self_evolution_cycle(&audit_signals);

    println!("\n🧠 Brain generated {} mercy-gated improvement proposals", proposals.len());

    if let Some(first_proposal) = proposals.first() {
        println!("   → Top proposal: {} (Mercy Impact: {:.2})", 
                 first_proposal.description, first_proposal.expected_mercy_impact);

        // 3. Apply the proposal using Hands (plasticity-engine-v2)
        println!("\n✋ Applying proposal via SafePlasticityApplicator...");
        match orchestrator.apply_improvement_proposal(first_proposal) {
            Ok(rollback_plan) => {
                println!("   ✅ Change applied successfully. Rollback available.");

                // 4. Verify the outcome
                println!("\n🔍 Verifying outcome...");
                let decision = orchestrator.verify_and_adapt(first_proposal, &rollback_plan);

                match decision {
                    VerificationDecision::Accept => println!("   ✅ Decision: ACCEPT — Improvement kept."),
                    VerificationDecision::Rollback => println!("   ↩️ Decision: ROLLBACK — Change reverted for safety."),
                    VerificationDecision::Reinforce => println!("   💪 Decision: REINFORCE — Further plasticity applied."),
                    VerificationDecision::FurtherAnalysis => println!("   🔬 Decision: FURTHER ANALYSIS needed."),
                }
            }
            Err(e) => println!("   ❌ Failed to apply proposal: {}", e),
        }
    }

    println!("\n✅ Self-evolution cycle complete.\n");
}