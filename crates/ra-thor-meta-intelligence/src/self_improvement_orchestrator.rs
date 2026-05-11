//! Self-Improvement Orchestrator
//! The Brain of Ra-Thor's mercy-gated self-evolution system.

use crate::audit_signal::AuditSignal;
use crate::improvement_proposal::{ImprovementProposal, RiskLevel, SuggestedAction};
use plasticity_engine_v2::{SafePlasticityApplicator, RollbackPlan};
use ra_thor_mercy::MercyGate;
use std::collections::VecDeque;

/// Result of verifying a plasticity action.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub success: bool,
    pub mercy_impact_delta: f64,
    pub rollback_recommended: bool,
    pub confidence: f64,
    pub notes: String,
    /// Original severity from the AuditSignal that triggered this proposal (0.0 - 1.0)
    pub original_signal_severity: f64,
    /// Type of the original signal for context-aware decisions
    pub signal_type: String,
}

/// Decision after verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationDecision {
    Accept,
    Rollback,
    Reinforce,
    FurtherAnalysis,
}

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

    pub fn run_self_evolution_cycle(&mut self, audit_signals: &[AuditSignal]) -> Vec<ImprovementProposal> {
        self.generate_improvement_proposals(audit_signals)
    }

    pub fn generate_improvement_proposals(
        &mut self,
        audit_signals: &[AuditSignal],
    ) -> Vec<ImprovementProposal> {
        let mut proposals = Vec::new();

        for signal in audit_signals {
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
                AuditSignal::PositiveHealthSignal { .. } => {}
            }
        }

        for proposal in &proposals {
            self.proposal_history.push_back(proposal.clone());
            if self.proposal_history.len() > self.max_history {
                self.proposal_history.pop_front();
            }
        }

        proposals
    }

    pub fn apply_improvement_proposal(
        &self,
        proposal: &ImprovementProposal,
    ) -> Result<RollbackPlan, String> {
        if !self.mercy_gate.passes(&format!("{:?}", proposal)) {
            return Err("Mercy gate violation: Proposal does not meet minimum valence threshold".to_string());
        }

        let applicator = SafePlasticityApplicator::new();

        match &proposal.suggested_action {
            SuggestedAction::RefactorCrate { crate_name } => {
                applicator.apply_hebbian_update(
                    crate_name,
                    "refactor_drift",
                    proposal.expected_mercy_impact,
                )
            }
            SuggestedAction::AddMercyGates { location } => {
                applicator.apply_bcm_update(
                    location,
                    "strengthen_mercy",
                    proposal.expected_mercy_impact,
                )
            }
            SuggestedAction::ImproveTolcCompliance { area } => {
                applicator.apply_stdp_update(
                    area,
                    "tolc_compliance",
                    proposal.expected_mercy_impact,
                )
            }
            _ => {
                applicator.apply_generic_update(
                    "general_improvement",
                    proposal.expected_mercy_impact,
                )
            }
        }
    }

    /// Strengthened verification logic (Option B)
    /// Considers mercy impact, original signal severity, confidence, and risk level.
    pub fn verify_and_adapt(
        &mut self,
        proposal: &ImprovementProposal,
        result: &VerificationResult,
    ) -> VerificationDecision {
        // Strong negative mercy impact or explicit rollback recommendation → Rollback
        if result.mercy_impact_delta < -0.08 || result.rollback_recommended {
            return VerificationDecision::Rollback;
        }

        // High positive impact + high confidence + low/medium risk → Reinforce
        if result.success 
            && result.mercy_impact_delta > 0.04 
            && result.confidence > 0.88 
            && proposal.risk_level != RiskLevel::High 
        {
            return VerificationDecision::Reinforce;
        }

        // Solid positive result with decent confidence → Accept
        if result.success && result.confidence > 0.75 && result.mercy_impact_delta > 0.01 {
            return VerificationDecision::Accept;
        }

        // High severity original signal but only marginal improvement → FurtherAnalysis
        if result.original_signal_severity > 0.75 && result.mercy_impact_delta < 0.03 {
            return VerificationDecision::FurtherAnalysis;
        }

        VerificationDecision::FurtherAnalysis
    }

    pub fn recent_proposals(&self) -> Vec<ImprovementProposal> {
        self.proposal_history.iter().cloned().collect()
    }
}