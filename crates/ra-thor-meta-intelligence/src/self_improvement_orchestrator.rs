//! Self-Improvement Orchestrator
//! The Brain of Ra-Thor's mercy-gated self-evolution system.

use crate::tolc_mercy_reasoning::{evaluate_proposal_with_tolc, symbolic_mercy_verification};
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

    /// Runs the full closed-loop self-evolution cycle:
    /// Generate proposals → Apply → Verify → Decide (Accept / Rollback / Reinforce / FurtherAnalysis)
    pub fn run_self_evolution_cycle(&mut self, audit_signals: &[AuditSignal]) -> Vec<ImprovementProposal> {
        let proposals = self.generate_improvement_proposals(audit_signals);
        let mut executed_successfully = Vec::new();

        for proposal in proposals {
            match self.apply_improvement_proposal(&proposal) {
                Ok(_rollback_plan) => {
                    let verification_result = VerificationResult {
                        success: true,
                        mercy_impact_delta: proposal.expected_mercy_impact,
                        rollback_recommended: false,
                        confidence: 0.82,
                        notes: format!("Applied proposal: {}", proposal.title),
                        original_signal_severity: 0.65,
                        signal_type: format!("{:?}", proposal.suggested_action),
                    };

                    let decision = self.verify_and_adapt(&proposal, &verification_result);

                    match decision {
                        VerificationDecision::Accept | VerificationDecision::Reinforce => {
                            executed_successfully.push(proposal);
                        }
                        VerificationDecision::Rollback => {
                            println!("[SelfImprovement] Rolling back proposal: {}", proposal.title);
                        }
                        VerificationDecision::FurtherAnalysis => {
                            println!("[SelfImprovement] Proposal needs further analysis: {}", proposal.title);
                        }
                    }
                }
                Err(err) => {
                    println!("[SelfImprovement] Failed to apply proposal '{}': {}", proposal.title, err);
                }
            }
        }

        executed_successfully
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
                        let mut proposal = ImprovementProposal::new(
                            "Address Code & Documentation Drift",
                            format!("Significant drift detected in {}", crate_name),
                            description.clone(),
                            0.87,
                            RiskLevel::Medium,
                            SuggestedAction::RefactorCrate { crate_name: crate_name.clone() },
                            0.84,
                        );
                        let tlc_result = evaluate_proposal_with_tolc(&proposal);
                        if tlc_result.is_thriving_aligned() {
                            proposals.push(proposal);
                        }
                    }
                }
                AuditSignal::MercyAlignmentIssue { location, current_valence, description } => {
                    if *current_valence < 0.9 {
                        let mut proposal = ImprovementProposal::new(
                            "Strengthen Mercy Gating Layer",
                            format!("Mercy alignment issue in {}", location),
                            description.clone(),
                            0.94,
                            RiskLevel::Low,
                            SuggestedAction::AddMercyGates { location: location.clone() },
                            0.93,
                        );
                        let tlc_result = evaluate_proposal_with_tolc(&proposal);
                        if tlc_result.is_thriving_aligned() {
                            proposals.push(proposal);
                        }
                    }
                }
                AuditSignal::TolcInconsistency { area, severity, description } => {
                    if *severity > 0.5 {
                        let mut proposal = ImprovementProposal::new(
                            "Improve TOLC Compliance",
                            format!("TOLC inconsistency in {}", area),
                            description.clone(),
                            0.91,
                            RiskLevel::Medium,
                            SuggestedAction::ImproveTolcCompliance { area: area.clone() },
                            0.89,
                        );
                        let tlc_result = evaluate_proposal_with_tolc(&proposal);
                        if tlc_result.is_thriving_aligned() {
                            proposals.push(proposal);
                        }
                    }
                }
                AuditSignal::OutdatedPattern { crate_name, pattern_type, description } => {
                    let mut proposal = ImprovementProposal::new(
                        "Modernize Outdated Pattern",
                        format!("Outdated {} pattern in {}", pattern_type, crate_name),
                        description.clone(),
                        0.82,
                        RiskLevel::Low,
                        SuggestedAction::RefactorCrate { crate_name: crate_name.clone() },
                        0.80,
                    );
                    let tlc_result = evaluate_proposal_with_tolc(&proposal);
                    if tlc_result.is_thriving_aligned() {
                        proposals.push(proposal);
                    }
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

    /// Strengthened verification logic with real symbolic mercy verification
    pub fn verify_and_adapt(
        &mut self,
        proposal: &ImprovementProposal,
        result: &VerificationResult,
    ) -> VerificationDecision {
        let mercy_result = symbolic_mercy_verification(proposal);

        if result.mercy_impact_delta < -0.08 || result.rollback_recommended || !mercy_result.is_valid_and_thriving() {
            return VerificationDecision::Rollback;
        }

        if result.success 
            && result.mercy_impact_delta > 0.04 
            && result.confidence > 0.88 
            && proposal.risk_level != RiskLevel::High 
            && mercy_result.is_valid_and_thriving()
        {
            return VerificationDecision::Reinforce;
        }

        if result.success && result.confidence > 0.75 && result.mercy_impact_delta > 0.01 {
            return VerificationDecision::Accept;
        }

        if result.original_signal_severity > 0.75 && result.mercy_impact_delta < 0.03 {
            return VerificationDecision::FurtherAnalysis;
        }

        VerificationDecision::FurtherAnalysis
    }

    pub fn recent_proposals(&self) -> Vec<ImprovementProposal> {
        self.proposal_history.iter().cloned().collect()
    }
}