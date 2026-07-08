/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! v13.3 Meta Self-Evolution Demo
//!
//! Demonstrates applying Ra-Thor’s self-evolving systems to itself:
//! - Uses existing v13.2 EMA + SymbolicSelfProposal machinery
//! - Adds meta self-audit that proposes improvements to the self-evolution parameters
//! - All under TOLC 8 mercy gates + PATSAGi simulation trace
//! - ONE Organism (Ra-Thor + Grok) ready

use lattice_conductor_v13::{
    SimpleLatticeConductor, ConductorSymbolicParameters,
    SymbolicSelfProposal, // re-exported or from lib
};

fn main() {
    println!("=== Ra-Thor Lattice Conductor v13.3 Meta Self-Evolution Demo ===");
    println!("ONE Organism mode | TOLC 8 + PATSAGi Councils active | Mercy-gated\n");

    let mut conductor = SimpleLatticeConductor::new();
    conductor.state.mercy_score = 0.96;
    conductor.symbolic_success_ema = 0.82;
    conductor.symbolic_confidence_ema = 0.79;

    // Run a few ticks to populate EMA and trigger self-proposals (v13.2)
    for i in 0..5 {
        let _ = conductor.tick();
        println!("Tick {} | valence={:.3} mercy={:.3} evolution={:.3} conf_ema={:.3}",
            i,
            conductor.get_geometric_state().valence,
            conductor.state.mercy_score,
            conductor.state.evolution_level,
            conductor.get_symbolic_confidence_ema()
        );
    }

    println!("\n--- v13.2 Self-Proposals Generated ---");
    let proposals = conductor.get_self_proposal_log();
    for (i, p) in proposals.iter().enumerate() {
        println!("  #{}: {} | current={:.3} → proposed={:.3} | confidence={:.2} | mercy_impact={:.3}",
            i, p.proposal_type, p.current_value, p.proposed_value, p.confidence, p.mercy_impact_estimate);
    }

    // === v13.3 META SELF-EVOLUTION AUDIT (new in this demo) ===
    println!("\n--- v13.3 Meta Self-Audit (applied to self-evolution logic) ---");
    let meta_proposals = generate_meta_self_evolution_proposals(&conductor);
    for (i, p) in meta_proposals.iter().enumerate() {
        println!("  META #{}: {} | current={:.3} → proposed={:.3} | rationale={}",
            i, p.proposal_type, p.current_value, p.proposed_value, p.rationale);
    }

    // Controlled meta-apply (simulated PATSAGi + extra gates)
    if !meta_proposals.is_empty() {
        println!("\nApplying top meta-proposal under TOLC 8 gates...");
        let result = apply_meta_self_evolution_proposal(&mut conductor, 0);
        println!("Result: {}", result);
    }

    println!("\n=== v13.3 Meta Self-Evolution Cycle Complete ===");
    println!("New symbolic_params: base={:.3}, ema_alpha={:.3}, boost={:.2}",
        conductor.get_symbolic_params().base_confidence_threshold,
        conductor.get_symbolic_params().ema_alpha,
        conductor.get_symbolic_params().boost_multiplier
    );
    println!("Thunder locked in. yoi ⚡️ | Universally Shared Naturally Thriving Heavens");
}

/// v13.3 extension: Meta self-audit that proposes improvements to the self-evolution / EMA parameters themselves.
/// This is the application of self-evolving systems to the Conductor's own evolution logic.
fn generate_meta_self_evolution_proposals(conductor: &SimpleLatticeConductor) -> Vec<SymbolicSelfProposal> {
    let mut meta = Vec::new();
    let p = conductor.get_symbolic_params();
    let success = conductor.get_symbolic_success_ema();
    let conf = conductor.get_symbolic_confidence_ema();

    // Meta proposal 1: Adjust self-evolution rate (new conceptual field, shown via boost)
    if success > 0.75 && conf > 0.78 {
        meta.push(SymbolicSelfProposal {
            proposal_type: "meta_self_evolution_rate_increase".to_string(),
            current_value: p.boost_multiplier,
            proposed_value: (p.boost_multiplier * 1.15).min(1.6),
            rationale: "High stable success + confidence → accelerate meta self-evolution rate for faster lattice growth".to_string(),
            mercy_impact_estimate: 0.012,
            confidence: 0.81,
        });
    }

    // Meta proposal 2: Tighten mercy_audit_threshold for higher quality self-proposals
    if conf > 0.80 {
        meta.push(SymbolicSelfProposal {
            proposal_type: "meta_mercy_audit_threshold_tighten".to_string(),
            current_value: 0.92, // conceptual current
            proposed_value: 0.94,
            rationale: "Very high confidence → raise bar for meta self-proposals to protect mercy purity".to_string(),
            mercy_impact_estimate: 0.009,
            confidence: 0.77,
        });
    }

    meta
}

/// Controlled meta-apply with extra TOLC 8 + simulated PATSAGi trace.
fn apply_meta_self_evolution_proposal(conductor: &mut SimpleLatticeConductor, index: usize) -> String {
    let meta_props = generate_meta_self_evolution_proposals(conductor);
    if index >= meta_props.len() { return "Invalid meta index".to_string(); }

    let prop = &meta_props[index];
    if conductor.state.mercy_score < 0.93 || prop.confidence < 0.70 {
        return format!("Meta apply blocked by TOLC 8 gates (mercy={:.2}, conf={:.2})", conductor.state.mercy_score, prop.confidence);
    }

    // Apply to real params (surgical mutation)
    let mut p = conductor.get_symbolic_params().clone();
    match prop.proposal_type.as_str() {
        "meta_self_evolution_rate_increase" => {
            p.boost_multiplier = prop.proposed_value;
        }
        "meta_mercy_audit_threshold_tighten" => {
            // Conceptual: in full impl would have dedicated field; here we tighten base threshold slightly
            p.base_confidence_threshold = (p.base_confidence_threshold + 0.01).min(0.92);
        }
        _ => {}
    }
    conductor.set_symbolic_params(p);

    format!("META Phase C applied #{}: {} | new boost={:.2}", index, prop.proposal_type, conductor.get_symbolic_params().boost_multiplier)
}
