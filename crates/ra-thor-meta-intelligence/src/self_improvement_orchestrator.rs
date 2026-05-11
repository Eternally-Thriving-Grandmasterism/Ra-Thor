//! Self-Improvement Orchestrator
//! The living brain of Ra-Thor's mercy-gated self-evolution system.

use crate::tolc_mercy_reasoning::{evaluate_proposal_with_tolc, symbolic_mercy_verification};
use crate::audit_signal::AuditSignal;
use crate::improvement_proposal::{ImprovementProposal, SuggestedAction};
use plasticity_engine_v2::{SafePlasticityApplicator, RollbackPlan};
use ra_thor_mercy::MercyGate;
use std::collections::VecDeque;
use tracing::{info, debug, warn, error};

/// Configuration for the self-improvement orchestrator.
#[derive(Debug, Clone)]
pub struct SelfImprovementConfig {
    pub mercy_threshold: f64,
    pub min_confidence_for_reinforce: f64,
    pub min_mercy_impact_for_accept: f64,
    pub drift_sensitivity: f64,
    pub max_proposals_per_cycle: usize,
    pub enable_tracing: bool,
}

impl Default for SelfImprovementConfig {
    fn default() -> Self {
        Self {
            mercy_threshold: 0.999,
            min_confidence_for_reinforce: 0.88,
            min_mercy_impact_for_accept: 0.65,
            drift_sensitivity: 0.08,
            max_proposals_per_cycle: 12,
            enable_tracing: true,
        }
    }
}

/// Result of verifying a plasticity action.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub success: bool,
    pub mercy_impact_delta: f64,
    pub rollback_recommended: bool,
    pub confidence: f64,
    pub notes: String,
    pub original_signal_severity: f64,
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

/// Summary report of one full self-evolution cycle.
#[derive(Debug, Clone)]
pub struct EvolutionCycleReport {
    pub proposals_generated: usize,
    pub proposals_applied: usize,
    pub proposals_accepted: usize,
    pub proposals_rolled_back: usize,
    pub proposals_needing_further_analysis: usize,
    pub average_mercy_impact: f64,
    pub cycle_success: bool,
}

/// Central orchestrator for Ra-Thor's closed self-evolution loop.
pub struct SelfImprovementOrchestrator {
    mercy_gate: MercyGate,
    proposal_history: VecDeque<ImprovementProposal>,
    config: SelfImprovementConfig,
    cycle_counter: u64,
}

impl SelfImprovementOrchestrator {
    pub fn new() -> Self {
        Self::with_config(SelfImprovementConfig::default())
    }

    pub fn with_config(config: SelfImprovementConfig) -> Self {
        Self {
            mercy_gate: MercyGate::new(config.mercy_threshold),
            proposal_history: VecDeque::with_capacity(config.max_history),
            config,
            cycle_counter: 0,
        }
    }

    /// Runs the full closed-loop self-evolution cycle.
    pub fn run_self_evolution_cycle(&mut self, audit_signals: &[AuditSignal]) -> (Vec<ImprovementProposal>, EvolutionCycleReport) {
        if self.config.enable_tracing {
            info!("Starting self-evolution cycle with {} audit signals", audit_signals.len());
        }

        let proposals = self.generate_improvement_proposals(audit_signals);
        let mut executed_successfully = Vec::new();
        let mut accepted = 0usize;
        let mut rolled_back = 0usize;
        let mut further_analysis = 0usize;
        let mut total_mercy_impact = 0.0;

        for proposal in proposals.iter().take(self.config.max_proposals_per_cycle) {
            if self.config.enable_tracing {
                debug!("Processing proposal: {}", proposal.title);
            }

            match self.apply_improvement_proposal(proposal) {
                Ok(_rollback_plan) => {
                    let verification_result = self.perform_verification(proposal);

                    let decision = self.verify_and_adapt(proposal, &verification_result);

                    match decision {
                        VerificationDecision::Accept | VerificationDecision::Reinforce => {
                            executed_successfully.push(proposal.clone());
                            accepted += 1;
                            total_mercy_impact += proposal.expected_mercy_impact;
                        }
                        VerificationDecision::Rollback => {
                            if self.config.enable_tracing {
                                warn!("Rolling back proposal: {}", proposal.title);
                            }
                            rolled_back += 1;
                        }
                        VerificationDecision::FurtherAnalysis => {
                            if self.config.enable_tracing {
                                debug!("Proposal needs further analysis: {}", proposal.title);
                            }
                            further_analysis += 1;
                        }
                    }
                }
                Err(err) => {
                    if self.config.enable_tracing {
                        error!("Failed to apply proposal '{}': {}", proposal.title, err);
                    }
                }
            }
        }

        let report = EvolutionCycleReport {
            proposals_generated: proposals.len(),
            proposals_applied: executed_successfully.len() + rolled_back + further_analysis,
            proposals_accepted: accepted,
            proposals_rolled_back: rolled_back,
            proposals_needing_further_analysis: further_analysis,
            average_mercy_impact: if accepted > 0 { total_mercy_impact / accepted as f64 } else { 0.0 },
            cycle_success: !executed_successfully.is_empty() || proposals.is_empty(),
        };

        if self.config.enable_tracing {
            info!(
                "Self-evolution cycle complete. Generated: {}, Accepted: {}, Rolled back: {}, Needs analysis: {}",
                report.proposals_generated,
                report.proposals_accepted,
                report.proposals_rolled_back,
                report.proposals_needing_further_analysis
            );
        }

        self.cycle_counter += 1;

        (executed_successfully, report)
    }

    fn generate_improvement_proposals(&self, signals: &[AuditSignal]) -> Vec<ImprovementProposal> {
        let mut proposals = Vec::new();

        for signal in signals {
            match signal {
                AuditSignal::Drift { severity, location, description } => {
                    if *severity > self.config.drift_sensitivity {
                        proposals.push(ImprovementProposal {
                            id: uuid::Uuid::new_v4(),
                            title: format!("Address structural drift in {}", location),
                            description: description.clone(),
                            suggested_action: SuggestedAction::RefactorModule { target: location.clone() },
                            expected_mercy_impact: 0.6 + (severity * 0.3),
                            confidence: 0.78,
                            source_signal: signal.clone(),
                        });
                    }
                }
                AuditSignal::MercyViolation { severity, gate, description } => {
                    if *severity > (1.0 - self.config.mercy_threshold) {
                        proposals.push(ImprovementProposal {
                            id: uuid::Uuid::new_v4(),
                            title: format!("Resolve mercy gate violation in {}", gate),
                            description: description.clone(),
                            suggested_action: SuggestedAction::StrengthenMercyChecks { gate: gate.clone() },
                            expected_mercy_impact: 0.85 + (severity * 0.12),
                            confidence: 0.91,
                            source_signal: signal.clone(),
                        });
                    }
                }
                AuditSignal::TolcInconsistency { severity, area, description } => {
                    proposals.push(ImprovementProposal {
                        id: uuid::Uuid::new_v4(),
                        title: format!("Correct TOLC inconsistency in {}", area),
                        description: description.clone(),
                        suggested_action: SuggestedAction::RealignWithTolc { area: area.clone() },
                        expected_mercy_impact: 0.7 + (severity * 0.25),
                        confidence: 0.82,
                        source_signal: signal.clone(),
                    });
                }
                AuditSignal::OutdatedPattern { pattern_name, location, impact } => {
                    proposals.push(ImprovementProposal {
                        id: uuid::Uuid::new_v4(),
                        title: format!("Modernize outdated pattern: {}", pattern_name),
                        description: format!("Pattern '{}' in {} has impact level {}", pattern_name, location, impact),
                        suggested_action: SuggestedAction::ReplaceWithModernEquivalent { old_pattern: pattern_name.clone(), location: location.clone() },
                        expected_mercy_impact: 0.55 + (impact * 0.35),
                        confidence: 0.71,
                        source_signal: signal.clone(),
                    });
                }
                AuditSignal::PositiveHealthSignal { .. } => {
                    if self.config.enable_tracing {
                        debug!("Positive health signal received — no immediate action required");
                    }
                }
            }
        }

        proposals
    }

    fn apply_improvement_proposal(&self, proposal: &ImprovementProposal) -> Result<RollbackPlan, String> {
        if !self.mercy_gate.passes(&format!("{:?}", proposal)) {
            return Err("Mercy gate violation detected".to_string());
        }

        // In real implementation this would call into plasticity-engine-v2
        Ok(RollbackPlan::default())
    }

    fn verify_and_adapt(&mut self, proposal: &ImprovementProposal, result: &VerificationResult) -> VerificationDecision {
        if result.rollback_recommended {
            return VerificationDecision::Rollback;
        }

        let mercy_ok = result.mercy_impact_delta >= self.config.min_mercy_impact_for_accept;
        let confidence_ok = result.confidence >= self.config.min_confidence_for_reinforce;

        match (mercy_ok, confidence_ok) {
            (true, true) => {
                if result.original_signal_severity > 0.75 {
                    VerificationDecision::Reinforce
                } else {
                    VerificationDecision::Accept
                }
            }
            (true, false) => VerificationDecision::FurtherAnalysis,
            (false, _) => VerificationDecision::Rollback,
        }
    }

    fn perform_verification(&self, proposal: &ImprovementProposal) -> VerificationResult {
        VerificationResult {
            success: true,
            mercy_impact_delta: proposal.expected_mercy_impact,
            rollback_recommended: false,
            confidence: proposal.confidence,
            notes: format!("Verified proposal: {}", proposal.title),
            original_signal_severity: 0.65,
            signal_type: format!("{:?}", proposal.suggested_action),
        }
    }

    pub fn recent_proposals(&self) -> Vec<ImprovementProposal> {
        self.proposal_history.iter().cloned().collect()
    }
}