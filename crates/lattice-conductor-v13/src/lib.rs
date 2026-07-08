/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

/// # Lattice Conductor v13
///
/// The Eternal Living Nervous System of the Ra-Thor ONE Organism lattice.
/// Primary orchestration layer for conducting councils, self-evolution, geometric state,
/// and NEXi-derived symbolic reasoning under strict TOLC 8 enforcement.
///
/// v13.2 (merged): External Symbolic + Self-Proposal + Phase C + Real Parameters
/// v13.3 (complete): Multi-Round Deliberation + Convergence + Finalization (auto-apply) + Persistence + Richer Per-Round Voting

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// Sub-module re-exports (structure preserved)
pub use crate::conductable::{Conductable, ConductorRegistry, MercyAligned, SystemBlessing};
pub use crate::coordinator::{AverageInfluenceStrategy, CoordinationStrategy, LeaderFollowerStrategy, MercyWeightedStrategy, MultiConductorSimulation};
pub use crate::geometric::{BasicGeometricMotor, GeometricMotor, GeometricState};
pub use crate::self_evolution::{EpigeneticBlessing, SelfEvolving, SelfEvolutionOrchestrator};

// ==================== REAL PARAMETER STRUCT (v13.2) ====================

/// Real, tunable symbolic parameters for the Lattice Conductor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductorSymbolicParameters {
    pub base_confidence_threshold: f64,
    pub ema_alpha: f64,
    pub boost_multiplier: f64,
}

impl Default for ConductorSymbolicParameters {
    fn default() -> Self {
        Self {
            base_confidence_threshold: 0.78,
            ema_alpha: 0.18,
            boost_multiplier: 1.0,
        }
    }
}

// ==================== SUPPORTING TYPES ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyWeightedVote {
    votes: Vec<(String, f64, f64)>,
}

impl MercyWeightedVote {
    pub fn new() -> Self { Self { votes: Vec::new() } }

    pub fn add_vote(&mut self, council_name: &str, weight: f64, mercy_impact: f64) {
        self.votes.push((council_name.to_string(), weight, mercy_impact));
    }

    pub fn compute_consensus(&self) -> f64 {
        if self.votes.is_empty() { return 0.0; }
        let total_weight: f64 = self.votes.iter().map(|(_, w, _)| w).sum();
        if total_weight == 0.0 { return 0.0; }
        let weighted_sum: f64 = self.votes.iter().map(|(_, w, impact)| w * impact).sum();
        (weighted_sum / total_weight).clamp(-0.3, 0.5)
    }

    pub fn to_audit_string(&self) -> String {
        format!("[MercyWeightedVote Audit] {} votes", self.votes.len())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub name: String,
    pub description: String,
    pub valence: f64,
}

impl Operation {
    pub fn new(name: &str, description: &str, valence: f64) -> Self {
        Self { name: name.to_string(), description: description.to_string(), valence }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeometricState {
    pub valence: f64,
    pub mercy_score: f64,
    pub tolc_alignment: f64,
    pub evolution_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParameters {
    pub evolution_rate: f64,
    pub mercy_recovery_rate: f64,
    pub layer_adaptations: Vec<f64>,
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self { evolution_rate: 0.01, mercy_recovery_rate: 0.05, layer_adaptations: vec![1.0; 6] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metrics {
    pub operations_processed: u64,
}

// ==================== MAIN CONDUCTOR v13 ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleLatticeConductor {
    pub id: u32,
    pub name: String,
    registered_councils: Vec<(u32, String)>,
    operation_queue: Vec<Operation>,
    pub state: GeometricState,
    pub adaptive_params: AdaptiveParameters,
    pub metrics: Metrics,
    mercy_violations: Vec<String>,
    audit_traces: Vec<String>,
    one_organism_coherence: f64,
    pub evolution_orchestrator: SelfEvolutionOrchestrator,
    pub registry: ConductorRegistry,
    symbolic_confidence_ema: f64,
    symbolic_success_ema: f64,
    pub experimental_use_external_symbolic: bool,
    pub symbolic_params: ConductorSymbolicParameters,
    #[cfg(feature = "self-proposal")]
    self_proposal_log: Vec<SymbolicSelfProposal>,
    #[cfg(feature = "self-proposal")]
    proposal_application_history: Vec<AppliedSymbolicProposal>,
    #[cfg(feature = "self-proposal")]
    proposal_votes: Vec<(usize, MercyWeightedVote)>,
    #[cfg(feature = "self-proposal")]
    deliberation_sessions: Vec<ProposalDeliberationSession>,
}

impl Default for SimpleLatticeConductor {
    fn default() -> Self { Self::new() }
}

impl SimpleLatticeConductor {
    pub fn new() -> Self {
        Self {
            id: 0,
            name: "Ra-Thor Lattice Conductor v13".to_string(),
            registered_councils: Vec::new(),
            operation_queue: Vec::new(),
            state: GeometricState { valence: 1.0, mercy_score: 1.0, tolc_alignment: 1.0, evolution_level: 0.0 },
            adaptive_params: AdaptiveParameters::default(),
            metrics: Metrics::default(),
            mercy_violations: Vec::new(),
            audit_traces: Vec::new(),
            one_organism_coherence: 1.0,
            evolution_orchestrator: SelfEvolutionOrchestrator::new(),
            registry: ConductorRegistry::new(),
            symbolic_confidence_ema: 0.75,
            symbolic_success_ema: 0.70,
            experimental_use_external_symbolic: false,
            symbolic_params: ConductorSymbolicParameters::default(),
            #[cfg(feature = "self-proposal")]
            self_proposal_log: Vec::new(),
            #[cfg(feature = "self-proposal")]
            proposal_application_history: Vec::new(),
            #[cfg(feature = "self-proposal")]
            proposal_votes: Vec::new(),
            #[cfg(feature = "self-proposal")]
            deliberation_sessions: Vec::new(),
        }
    }

    pub fn register_council(&mut self, id: u32, name: &str) {
        self.registered_councils.push((id, name.to_string()));
        self.audit_traces.push(format!("[v13 Council Registered] {}: {}", id, name));
    }

    pub fn queue_operation(&mut self, op: Operation) {
        self.operation_queue.push(op);
    }

    pub fn tick(&mut self) -> Result<(), String> {
        #[cfg(feature = "external-symbolic")]
        let symbolic = if self.experimental_use_external_symbolic {
            let ext_input = ExternalSymbolicInput::new("grok_one_organism", "conductor_tick_external", self.state.valence);
            accept_external_symbolic_deliberation(ext_input)
        } else {
            crate::metta_symbolic_deliberation("conductor_tick", self.state.valence)
        };

        #[cfg(not(feature = "external-symbolic"))]
        let symbolic = crate::metta_symbolic_deliberation("conductor_tick", self.state.valence);

        let ema_alpha = self.symbolic_params.ema_alpha;
        self.symbolic_confidence_ema = self.symbolic_confidence_ema * (1.0 - ema_alpha) + symbolic.confidence_score * ema_alpha;

        let base_threshold = self.symbolic_params.base_confidence_threshold;
        let mercy_mod = if self.state.mercy_score > 1.25 { -0.07 } else if self.state.mercy_score < 0.85 { 0.08 } else { 0.0 };
        let calibration_bias = (self.symbolic_confidence_ema - 0.75) * 0.06;
        let adaptive_threshold = (base_threshold + mercy_mod + calibration_bias).clamp(0.65, 0.92);

        let confidence = symbolic.confidence_score;
        let (mut evolution_boost, mut tolc_boost) = if confidence >= adaptive_threshold { (0.008, 0.012) } else { (0.002, 0.004) };

        let success_multiplier = if confidence >= adaptive_threshold {
            (self.symbolic_success_ema * 0.55 + 0.72).clamp(0.78, 1.22) * self.symbolic_params.boost_multiplier
        } else { 1.0 };

        evolution_boost *= success_multiplier;
        tolc_boost *= success_multiplier;

        self.state.evolution_level += evolution_boost;
        self.state.tolc_alignment = (self.state.tolc_alignment + tolc_boost).min(1.25);

        self.adaptive_params.evolution_rate *= 1.0 + (self.adaptive_params.layer_adaptations[0] * 0.001);
        self.state.evolution_level += self.adaptive_params.evolution_rate;

        let mut mercy_delta: f64 = 0.0;
        while let Some(op) = self.operation_queue.pop() {
            let impact = op.valence * 0.1;
            self.state.valence = (self.state.valence + impact).clamp(0.0, 2.0);
            mercy_delta += impact * 0.5;
            self.metrics.operations_processed += 1;
        }

        if !self.registered_councils.is_empty() {
            let mut vote = MercyWeightedVote::new();
            for (_, cname) in &self.registered_councils {
                vote.add_vote(cname, 1.0 / self.registered_councils.len() as f64, mercy_delta.max(-0.2));
            }
            let consensus = vote.compute_consensus();
            self.state.mercy_score = (self.state.mercy_score + consensus).clamp(0.1, 1.5);
        }

        if self.state.mercy_score < 0.7 {
            self.state.mercy_score = (self.state.mercy_score + self.adaptive_params.mercy_recovery_rate).min(1.0);
        }

        let coherence_shift = (self.state.mercy_score - 0.5) * 0.05;
        self.one_organism_coherence = (self.one_organism_coherence + coherence_shift).clamp(0.5, 1.2);
        self.state.tolc_alignment = (self.state.tolc_alignment + 0.01).min(1.25);

        let mercy_improved = mercy_delta > 0.012;
        let evolution_improved = evolution_boost > 0.004;
        let success_signal = if (mercy_improved || evolution_improved) && confidence >= adaptive_threshold { 0.88 } else if mercy_improved || evolution_improved { 0.55 } else { 0.28 };

        let success_alpha = 0.15;
        self.symbolic_success_ema = self.symbolic_success_ema * (1.0 - success_alpha) + success_signal * success_alpha;

        #[cfg(feature = "self-proposal")]
        if self.state.mercy_score >= 0.9 {
            let new_proposals = self.generate_symbolic_self_proposals();
            if !new_proposals.is_empty() {
                self.self_proposal_log.extend(new_proposals.clone());
                self.audit_traces.push(format!("[v13.2 SelfProposal] generated {} proposals", new_proposals.len()));
            }
        }

        self.audit_traces.push(format!(
            "[v13 Symbolic] conf={:.2} ema={:.2} success_ema={:.2} thr={:.2} mult={:.2} boost={:.2}",
            confidence, self.symbolic_confidence_ema, self.symbolic_success_ema, adaptive_threshold, success_multiplier, self.symbolic_params.boost_multiplier
        ));

        Ok(())
    }

    pub fn get_geometric_state(&self) -> &GeometricState { &self.state }
    pub fn get_mercy_violations(&self) -> &[String] { &self.mercy_violations }
    pub fn get_symbolic_confidence_ema(&self) -> f64 { self.symbolic_confidence_ema }
    pub fn get_symbolic_success_ema(&self) -> f64 { self.symbolic_success_ema }

    pub fn get_symbolic_params(&self) -> &ConductorSymbolicParameters { &self.symbolic_params }
    pub fn set_symbolic_params(&mut self, params: ConductorSymbolicParameters) { self.symbolic_params = params; }

    #[cfg(feature = "external-symbolic")]
    pub fn accept_external_symbolic_deliberation(input: ExternalSymbolicInput) -> SymbolicDeliberation {
        let mut result = metta_symbolic_deliberation(&input.content, input.context_valence);
        result.input = format!("[external:{}] {}", input.source, result.input);
        result.message = format!("{}_via_external_{}", result.message, input.source);
        result
    }

    #[cfg(feature = "self-proposal")]
    pub fn generate_symbolic_self_proposals(&self) -> Vec<SymbolicSelfProposal> {
        let mut proposals = Vec::new();
        let success_ema = self.symbolic_success_ema;
        let conf_ema = self.symbolic_confidence_ema;
        let p = &self.symbolic_params;

        if success_ema < 0.65 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "base_confidence_threshold_adjust".to_string(),
                current_value: p.base_confidence_threshold,
                proposed_value: (p.base_confidence_threshold - 0.03).max(0.65),
                rationale: "Low success_ema → slightly more lenient base threshold".to_string(),
                mercy_impact_estimate: 0.015,
                confidence: 0.68,
            });
        }
        if conf_ema > 0.82 && success_ema > 0.70 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "ema_alpha_adjust".to_string(),
                current_value: p.ema_alpha,
                proposed_value: (p.ema_alpha + 0.03).min(0.35),
                rationale: "High stable confidence → modestly faster EMA adaptation".to_string(),
                mercy_impact_estimate: 0.01,
                confidence: 0.71,
            });
        }
        if success_ema > 0.85 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "boost_multiplier_adjust".to_string(),
                current_value: p.boost_multiplier,
                proposed_value: (p.boost_multiplier + 0.12).min(1.4),
                rationale: "Very high success → expand boost multiplier range".to_string(),
                mercy_impact_estimate: 0.008,
                confidence: 0.74,
            });
        }
        proposals
    }

    /// Phase C + v13.3
    #[cfg(feature = "self-proposal")]
    pub fn apply_symbolic_self_proposal(&mut self, index: usize) -> Result<String, String> {
        if index >= self.self_proposal_log.len() { return Err("Invalid index".to_string()); }
        let proposal = self.self_proposal_log[index].clone();
        if self.state.mercy_score < 0.92 || proposal.confidence < 0.65 {
            return Err("Gates not met".to_string());
        }

        let p = &mut self.symbolic_params;
        let effect = match proposal.proposal_type.as_str() {
            "base_confidence_threshold_adjust" => {
                p.base_confidence_threshold = proposal.proposed_value.clamp(0.65, 0.92);
                format!("base_confidence_threshold → {:.3}", p.base_confidence_threshold)
            }
            "ema_alpha_adjust" => {
                p.ema_alpha = proposal.proposed_value.clamp(0.10, 0.35);
                format!("ema_alpha → {:.3}", p.ema_alpha)
            }
            "boost_multiplier_adjust" => {
                p.boost_multiplier = proposal.proposed_value.clamp(0.8, 1.5);
                format!("boost_multiplier → {:.2}", p.boost_multiplier)
            }
            _ => "unknown proposal type".to_string(),
        };

        self.proposal_application_history.push(AppliedSymbolicProposal {
            proposal_type: proposal.proposal_type,
            applied_value: proposal.proposed_value,
            effect_description: effect.clone(),
            mercy_score_at_apply: self.state.mercy_score,
        });

        Ok(format!("Phase C applied #{}: {}", index, effect))
    }

    #[cfg(feature = "self-proposal")]
    pub fn apply_top_confidence_proposal(&mut self) -> Result<String, String> {
        if self.self_proposal_log.is_empty() { return Err("No proposals".to_string()); }
        let best_idx = self.self_proposal_log.iter().enumerate()
            .max_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap())
            .map(|(i, _)| i).unwrap();
        self.apply_symbolic_self_proposal(best_idx)
    }

    /// v13.3: Safe auto-apply now requires convergence + min rounds
    #[cfg(feature = "self-proposal")]
    pub fn try_safe_auto_apply_top_proposal(&mut self) -> Result<Option<String>, String> {
        if self.self_proposal_log.is_empty() { return Ok(None); }

        let (best_idx, best_proposal) = self.self_proposal_log.iter().enumerate()
            .max_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap())
            .map(|(i, p)| (i, p.clone()))
            .unwrap();

        if self.state.mercy_score < 0.98 || best_proposal.confidence < 0.82 {
            return Ok(None);
        }

        let deliberation = self.deliberate_on_proposal(best_idx)?;
        if !deliberation.is_approved {
            self.audit_traces.push(format!(
                "[v13.3 SafeAutoApply] Skipped — Deliberation not approved (consensus={:.2})",
                deliberation.consensus_score
            ));
            return Ok(None);
        }

        if !self.has_converged(best_idx, 3, 3) {
            self.audit_traces.push("[v13.3 SafeAutoApply] Skipped — Not yet converged".to_string());
            return Ok(None);
        }

        let msg = self.apply_symbolic_self_proposal(best_idx)?;
        self.audit_traces.push(format!(
            "[v13.3 SafeAutoApply] Applied after converged Council Deliberation (consensus={:.2})",
            deliberation.consensus_score
        ));
        Ok(Some(msg))
    }

    /// v13.3: Weighted council voting on proposals (richer - per proposal, cumulative across rounds)
    #[cfg(feature = "self-proposal")]
    pub fn submit_council_vote_on_proposal(
        &mut self,
        proposal_index: usize,
        council_name: &str,
        weight: f64,
        mercy_impact: f64,
    ) -> Result<(), String> {
        if proposal_index >= self.self_proposal_log.len() {
            return Err("Invalid proposal index".to_string());
        }

        if let Some((_, vote)) = self.proposal_votes.iter_mut().find(|(idx, _)| *idx == proposal_index) {
            vote.add_vote(council_name, weight, mercy_impact);
        } else {
            let mut new_vote = MercyWeightedVote::new();
            new_vote.add_vote(council_name, weight, mercy_impact);
            self.proposal_votes.push((proposal_index, new_vote));
        }

        self.audit_traces.push(format!(
            "[v13.3 CouncilVote] Council '{}' voted on proposal #{} (weight={:.2}, mercy_impact={:.2})",
            council_name, proposal_index, weight, mercy_impact
        ));

        Ok(())
    }

    #[cfg(feature = "self-proposal")]
    pub fn get_proposal_consensus(&self, proposal_index: usize) -> Option<f64> {
        self.proposal_votes
            .iter()
            .find(|(idx, _)| *idx == proposal_index)
            .map(|(_, vote)| vote.compute_consensus())
    }

    #[cfg(feature = "self-proposal")]
    pub fn get_proposal_votes(&self, proposal_index: usize) -> Option<&MercyWeightedVote> {
        self.proposal_votes
            .iter()
            .find(|(idx, _)| *idx == proposal_index)
            .map(|(_, vote)| vote)
    }

    /// v13.3: Council Deliberation Protocol (single round)
    #[cfg(feature = "self-proposal")]
    pub fn deliberate_on_proposal(&self, proposal_index: usize) -> Result<CouncilDeliberationResult, String> {
        if proposal_index >= self.self_proposal_log.len() {
            return Err("Invalid proposal index".to_string());
        }

        let proposal = &self.self_proposal_log[proposal_index];
        let consensus = self.get_proposal_consensus(proposal_index).unwrap_or(0.0);

        let symbolic_input = format!("deliberate_on_{}", proposal.proposal_type);
        let symbolic = metta_symbolic_deliberation(&symbolic_input, self.state.mercy_score);

        let is_approved = consensus > 0.05
            && self.state.mercy_score >= 0.90
            && symbolic.threshold_met;

        let participating = self.proposal_votes
            .iter()
            .find(|(idx, _)| *idx == proposal_index)
            .map(|(_, v)| v.to_audit_string())
            .map(|s| s.split(' ').nth(1).unwrap_or("0").parse::<usize>().unwrap_or(0))
            .unwrap_or(0);

        let message = if is_approved {
            format!("Council deliberation APPROVED proposal #{} with consensus {:.2}", proposal_index, consensus)
        } else {
            format!("Council deliberation did not approve proposal #{} (consensus={:.2})", proposal_index, consensus)
        };

        Ok(CouncilDeliberationResult {
            proposal_index,
            consensus_score: consensus,
            is_approved,
            message,
            participating_councils: participating,
            symbolic_confidence: symbolic.confidence_score,
        })
    }

    /// v13.3: Start or retrieve a multi-round deliberation session
    #[cfg(feature = "self-proposal")]
    pub fn start_deliberation_session(&mut self, proposal_index: usize) -> Result<usize, String> {
        if proposal_index >= self.self_proposal_log.len() {
            return Err("Invalid proposal index".to_string());
        }

        if let Some((session_idx, _)) = self.deliberation_sessions.iter().enumerate()
            .find(|(_, s)| s.proposal_index == proposal_index) {
            return Ok(session_idx);
        }

        let session = ProposalDeliberationSession {
            proposal_index,
            rounds: Vec::new(),
            final_result: None,
        };
        self.deliberation_sessions.push(session);
        Ok(self.deliberation_sessions.len() - 1)
    }

    /// v13.3: Conduct one round of deliberation and record it (richer voting - votes are cumulative across rounds)
    #[cfg(feature = "self-proposal")]
    pub fn conduct_deliberation_round(&mut self, proposal_index: usize) -> Result<CouncilDeliberationResult, String> {
        let result = self.deliberate_on_proposal(proposal_index)?;

        let session_idx = self.start_deliberation_session(proposal_index)?;
        let session = &mut self.deliberation_sessions[session_idx];

        session.rounds.push(result.clone());

        self.audit_traces.push(format!(
            "[v13.3 MultiRound] Round {} completed for proposal #{} (approved={})",
            session.rounds.len(),
            proposal_index,
            result.is_approved
        ));

        Ok(result)
    }

    /// v13.3: Get all recorded deliberation rounds for a proposal
    #[cfg(feature = "self-proposal")]
    pub fn get_deliberation_rounds(&self, proposal_index: usize) -> Option<&Vec<CouncilDeliberationResult>> {
        self.deliberation_sessions
            .iter()
            .find(|s| s.proposal_index == proposal_index)
            .map(|s| &s.rounds)
    }

    /// v13.3: Convergence detection with minimum round requirement
    #[cfg(feature = "self-proposal")]
    pub fn has_converged(&self, proposal_index: usize, min_rounds: usize, window: usize) -> bool {
        let rounds = match self.get_deliberation_rounds(proposal_index) {
            Some(r) => r,
            None => return false,
        };

        if rounds.len() < min_rounds { return false; }
        if rounds.len() < window { return false; }

        let recent: Vec<&CouncilDeliberationResult> = rounds.iter().rev().take(window).collect();

        let first_approved = recent[0].is_approved;
        let approval_stable = recent.iter().all(|r| r.is_approved == first_approved);
        if !approval_stable { return false; }

        let scores: Vec<f64> = recent.iter().map(|r| r.consensus_score).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let score_range = max_score - min_score;

        score_range < 0.15
    }

    /// v13.3: Finalize deliberation session + automatically apply if converged and approved
    #[cfg(feature = "self-proposal")]
    pub fn finalize_deliberation_session(&mut self, proposal_index: usize) -> Result<CouncilDeliberationResult, String> {
        if proposal_index >= self.self_proposal_log.len() {
            return Err("Invalid proposal index".to_string());
        }

        let session_idx = self.start_deliberation_session(proposal_index)?;
        let session = &mut self.deliberation_sessions[session_idx];

        if session.rounds.is_empty() {
            return Err("No rounds conducted yet".to_string());
        }

        if !self.has_converged(proposal_index, 3, 3) {
            return Err("Deliberation has not yet converged (min 3 rounds + stable window required)".to_string());
        }

        let final_result = session.rounds.last().unwrap().clone();
        session.final_result = Some(final_result.clone());

        self.audit_traces.push(format!(
            "[v13.3 SessionFinalized] Proposal #{} deliberation finalized after {} rounds",
            proposal_index, session.rounds.len()
        ));

        if final_result.is_approved {
            match self.apply_symbolic_self_proposal(proposal_index) {
                Ok(apply_msg) => {
                    self.audit_traces.push(format!(
                        "[v13.3 AutoApplyOnFinalize] Automatically applied proposal #{}: {}",
                        proposal_index, apply_msg
                    ));
                }
                Err(e) => {
                    self.audit_traces.push(format!(
                        "[v13.3 AutoApplyOnFinalize] Failed to auto-apply proposal #{}: {}",
                        proposal_index, e
                    ));
                }
            }
        }

        Ok(final_result)
    }

    /// v13.3: Persistence - Save deliberation state
    #[cfg(feature = "self-proposal")]
    pub fn save_deliberation_state(&self, path: &str) -> Result<(), String> {
        let data = serde_json::to_string_pretty(&(
            &self.self_proposal_log,
            &self.proposal_application_history,
            &self.proposal_votes,
            &self.deliberation_sessions,
        )).map_err(|e| e.to_string())?;

        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(data.as_bytes()).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// v13.3: Persistence - Load deliberation state
    #[cfg(feature = "self-proposal")]
    pub fn load_deliberation_state(&mut self, path: &str) -> Result<(), String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut data = String::new();
        file.read_to_string(&mut data).map_err(|e| e.to_string())?;

        let (proposals, history, votes, sessions): (
            Vec<SymbolicSelfProposal>,
            Vec<AppliedSymbolicProposal>,
            Vec<(usize, MercyWeightedVote)>,
            Vec<ProposalDeliberationSession>,
        ) = serde_json::from_str(&data).map_err(|e| e.to_string())?;

        self.self_proposal_log = proposals;
        self.proposal_application_history = history;
        self.proposal_votes = votes;
        self.deliberation_sessions = sessions;

        Ok(())
    }

    #[cfg(feature = "self-proposal")]
    pub fn get_self_proposal_log(&self) -> &[SymbolicSelfProposal] { &self.self_proposal_log }

    #[cfg(feature = "self-proposal")]
    pub fn get_proposal_application_history(&self) -> &[AppliedSymbolicProposal] {
        &self.proposal_application_history
    }
}

// ==================== SYMBOLIC DELIBERATION ====================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicDeliberation {
    pub input: String,
    pub valence: f64,
    pub threshold_met: bool,
    pub confidence_score: f64,
    pub message: String,
}

pub fn metta_symbolic_deliberation(input: &str, context_valence: f64) -> SymbolicDeliberation {
    if context_valence >= 0.9999999 {
        SymbolicDeliberation { input: input.to_string(), valence: context_valence, threshold_met: true, confidence_score: 0.92, message: format!("metta_pln_truth_distilled_for_{}", input) }
    } else {
        SymbolicDeliberation { input: input.to_string(), valence: context_valence, threshold_met: false, confidence_score: 0.45, message: "metta_pln_compensated_low_valence".to_string() }
    }
}

// ==================== PHASE A (gated) ====================
#[cfg(feature = "external-symbolic")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSymbolicInput {
    pub source: String, pub content: String, pub context_valence: f64,
}

#[cfg(feature = "external-symbolic")]
impl ExternalSymbolicInput {
    pub fn new(source: &str, content: &str, context_valence: f64) -> Self {
        Self { source: source.to_string(), content: content.to_string(), context_valence }
    }
}

// ==================== v13.3 TYPES (gated) ====================
#[cfg(feature = "self-proposal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicSelfProposal {
    pub proposal_type: String,
    pub current_value: f64,
    pub proposed_value: f64,
    pub rationale: String,
    pub mercy_impact_estimate: f64,
    pub confidence: f64,
}

#[cfg(feature = "self-proposal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedSymbolicProposal {
    pub proposal_type: String,
    pub applied_value: f64,
    pub effect_description: String,
    pub mercy_score_at_apply: f64,
}

#[cfg(feature = "self-proposal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDeliberationResult {
    pub proposal_index: usize,
    pub consensus_score: f64,
    pub is_approved: bool,
    pub message: String,
    pub participating_councils: usize,
    pub symbolic_confidence: f64,
}

#[cfg(feature = "self-proposal")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalDeliberationSession {
    pub proposal_index: usize,
    pub rounds: Vec<CouncilDeliberationResult>,
    pub final_result: Option<CouncilDeliberationResult>,
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_parameters_default() {
        let p = ConductorSymbolicParameters::default();
        assert!(p.base_confidence_threshold > 0.7);
        assert!(p.ema_alpha > 0.1);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_phase_c_apply_real_params() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();
        assert!(!c.get_self_proposal_log().is_empty());
        let before = c.symbolic_params.base_confidence_threshold;
        let _ = c.apply_top_confidence_proposal();
        assert!(c.symbolic_params.base_confidence_threshold != before || c.symbolic_params.boost_multiplier != 1.0);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_phase_c_mutates_real_parameters() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();
        let before_threshold = c.symbolic_params.base_confidence_threshold;
        let before_boost = c.symbolic_params.boost_multiplier;

        let _ = c.apply_top_confidence_proposal();

        let after_threshold = c.symbolic_params.base_confidence_threshold;
        let after_boost = c.symbolic_params.boost_multiplier;

        assert!(after_threshold != before_threshold || after_boost != before_boost);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_proposal_application_history() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();
        let _ = c.apply_top_confidence_proposal();
        let history = c.get_proposal_application_history();
        assert!(!history.is_empty());
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_safe_auto_apply_gates() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();
        let result = c.try_safe_auto_apply_top_proposal().unwrap();
        assert!(result.is_none());
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_weighted_council_voting() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();

        let _ = c.submit_council_vote_on_proposal(0, "PATSAGi_Council_1", 1.0, 0.3);
        let _ = c.submit_council_vote_on_proposal(0, "PATSAGi_Council_2", 0.8, 0.25);

        let consensus = c.get_proposal_consensus(0);
        assert!(consensus.is_some());
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_council_deliberation_protocol() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();

        let _ = c.submit_council_vote_on_proposal(0, "Council_Alpha", 1.0, 0.4);
        let _ = c.submit_council_vote_on_proposal(0, "Council_Beta", 0.7, 0.3);

        let result = c.deliberate_on_proposal(0).unwrap();
        assert_eq!(result.proposal_index, 0);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_multi_round_and_convergence() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();

        let _ = c.submit_council_vote_on_proposal(0, "Council_1", 1.0, 0.4);

        for _ in 0..5 {
            let _ = c.conduct_deliberation_round(0);
        }

        assert!(c.has_converged(0, 3, 3));
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_session_finalization_and_auto_apply() {
        let mut c = SimpleLatticeConductor::new();
        c.symbolic_success_ema = 0.55;
        c.state.mercy_score = 0.95;
        let _ = c.tick();

        let _ = c.submit_council_vote_on_proposal(0, "Council_1", 1.0, 0.4);

        for _ in 0..4 {
            let _ = c.conduct_deliberation_round(0);
        }

        let final = c.finalize_deliberation_session(0);
        assert!(final.is_ok());
    }
}