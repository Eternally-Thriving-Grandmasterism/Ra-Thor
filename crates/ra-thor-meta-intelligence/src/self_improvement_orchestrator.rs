//! Self-Improvement Orchestrator
//! The Brain of Ra-Thor's mercy-gated self-evolution system.

use crate::audit_signal::AuditSignal;
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

    /// Generates high-quality, mercy-gated Improvement Proposals from structured audit signals.
    /// This is the core decision function that connects ra-thor-monorepo-auditor to self-evolution.
    pub fn generate_improvement_proposals(
        &mut self,
        audit_signals: &[AuditSignal],
    ) -> Vec<ImprovementProposal> {
        let mut proposals = Vec::new();

        for signal in audit_signals {
            // Strict mercy gate - only process signals that pass high valence threshold
            if !self.mercy_gate.passes(&format!("{:?}", signal)) {
                continue;
            }

            match signal {
                AuditSignal::DriftDetected { crate_name, severity, description } => {
                    if *severity > 0.6 {
                        proposals.push(ImprovementProposal::new(
                            "Address Code & Documentation Drift",
                            format!("Significant drift detected in {}", crate_name),
                            description.clone(),
                            0.87,
                            RiskLevel::Medium,
                            SuggestedAction::RefactorCrate { crate_name: crate_name.clone() },
                            0.84,
                        ));
                    }
                }
                AuditSignal::MercyAlignmentIssue { location, current_valence, description } => {
                    if *current_valence < 0.9 {
                        proposals.push(ImprovementProposal::new(
                            "Strengthen Mercy Gating Layer",
                            format!("Mercy alignment issue in {}", location),
                            description.clone(),
                            0.94,
                            RiskLevel::Low,
                            SuggestedAction::AddMercyGates { location: location.clone() },
                            0.93,
                        ));
                    }
                }
                AuditSignal::TolcInconsistency { area, severity, description } => {
                    if *severity > 0.5 {
                        proposals.push(ImprovementProposal::new(
                            "Improve TOLC Compliance",
                            format!("TOLC inconsistency in {}", area),
                            description.clone(),
                            0.91,
                            RiskLevel::Medium,
                            SuggestedAction::ImproveTolcCompliance { area: area.clone() },
                            0.89,
                        ));
                    }
                }
                AuditSignal::OutdatedPattern { crate_name, pattern_type, description } => {
                    proposals.push(ImprovementProposal::new(
                        "Modernize Outdated Pattern",
                        format!("Outdated {} pattern in {}", pattern_type, crate_name),
                        description.clone(),
                        0.82,
                        RiskLevel::Low,
                        SuggestedAction::RefactorCrate { crate_name: crate_name.clone() },
                        0.80,
                    ));
                }
                AuditSignal::PositiveHealthSignal { .. } => {
                    // Positive signals can still generate reinforcement proposals
                    // but with lower priority
                }
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
