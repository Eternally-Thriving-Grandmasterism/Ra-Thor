//! Self-Improvement Orchestrator
//! The Brain of Ra-Thor's mercy-gated self-evolution system.

use crate::improvement_proposal::{ImprovementProposal, RiskLevel, SuggestedAction};
use ra_thor_mercy::MercyGate;
use std::collections::VecDeque;

/// Central orchestrator for Ra-Thor's closed self-evolution loop.
pub struct SelfImprovementOrchestrator {
    mercy_gate: MercyGate,
    proposal_history: VecDeque<ImprovementProposal>,
    max_history: usize,
}

impl SelfImprovementOrchestrator {
    pub fn new() -> Self {
        Self {
            mercy_gate: MercyGate::new(0.999),
            proposal_history: VecDeque::with_capacity(64),
            max_history: 64,
        }
    }

    /// Generates high-quality, mercy-gated Improvement Proposals.
    /// This is the core decision function of the self-evolution triad.
    pub fn generate_improvement_proposals(
        &mut self,
        audit_signals: &[String],
    ) -> Vec<ImprovementProposal> {
        let mut proposals = Vec::new();

        for signal in audit_signals {
            // Strict mercy gate — only high-valence signals are considered
            if !self.mercy_gate.passes(signal) {
                continue;
            }

            // TOLC + mercy-aware proposal generation
            if signal.contains("drift") || signal.contains("outdated") {
                proposals.push(ImprovementProposal::new(
                    "Address Code & Documentation Drift",
                    format!("Detected significant drift in: {}", signal),
                    "Outdated patterns reduce long-term maintainability, increase hallucination risk, and weaken mercy alignment.",
                    0.87,
                    RiskLevel::Medium,
                    SuggestedAction::RefactorCrate { crate_name: signal.clone() },
                    0.84,
                ));
            }

            if signal.contains("mercy") || signal.contains("low-valence") || signal.contains("weak gate") {
                proposals.push(ImprovementProposal::new(
                    "Strengthen Mercy Gating Layer",
                    format!("Mercy alignment issue detected: {}", signal),
                    "Strengthening mercy gates directly increases system safety, positive emotion propagation, and long-term thriving potential.",
                    0.94,
                    RiskLevel::Low,
                    SuggestedAction::AddMercyGates { location: signal.clone() },
                    0.93,
                ));
            }

            if signal.contains("TOLC") || signal.contains("inconsistent") {
                proposals.push(ImprovementProposal::new(
                    "Improve TOLC Compliance",
                    format!("TOLC inconsistency found: {}", signal),
                    "TOLC alignment is foundational to ethical self-evolution and long-term lattice coherence.",
                    0.91,
                    RiskLevel::Medium,
                    SuggestedAction::ImproveTolcCompliance { area: signal.clone() },
                    0.89,
                ));
            }
        }

        // Maintain bounded history
        for proposal in &proposals {
            self.proposal_history.push_back(proposal.clone());
            if self.proposal_history.len() > self.max_history {
                self.proposal_history.pop_front();
            }
        }

        proposals
    }

    pub fn recent_proposals(&self) -> Vec<ImprovementProposal> {
        self.proposal_history.iter().cloned().collect()
    }
}
