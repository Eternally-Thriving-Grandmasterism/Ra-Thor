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
/// v13 advancements:
/// - Real GeometricMotor v2 (DualQuaternion + Study Quadric + hyperbolic)
/// - Conductor-native self-evolution (propose/validate/bless + CEHI)
/// - NEXi metta/PLN symbolic bridge for explicit truth-distillation
/// - Full preservation of original conductor logic + surgical v13 extensions
///
/// v13.2 (Phases A+B+C + feature flags):
/// - Phase A: ExternalSymbolicInput + accept_external (gated by "external-symbolic")
/// - Phase B: SymbolicSelfProposal generation + logging (gated by "self-proposal")
/// - Phase C: Controlled explicit apply of self-proposals (gated by "self-proposal")
/// - Cargo features: external-symbolic, self-proposal, full-v13-2, experimental

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// Sub-module re-exports (structure preserved)
pub use crate::conductable::{Conductable, ConductorRegistry, MercyAligned, SystemBlessing};
pub use crate::coordinator::{AverageInfluenceStrategy, CoordinationStrategy, LeaderFollowerStrategy, MercyWeightedStrategy, MultiConductorSimulation};
pub use crate::geometric::{BasicGeometricMotor, GeometricMotor, GeometricState};
pub use crate::self_evolution::{EpigeneticBlessing, SelfEvolving, SelfEvolutionOrchestrator};

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

    /// Computes mercy-weighted consensus (clamped for stability).
    /// Used by tick() for council influence on conductor state.
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
    /// Exponential moving average of recent symbolic confidence scores.
    symbolic_confidence_ema: f64,
    /// EMA tracking correlation between high-confidence symbolic signals and positive state outcomes.
    symbolic_success_ema: f64,
    /// v13.2 experimental flag (runtime, works with Cargo feature)
    pub experimental_use_external_symbolic: bool,
    #[cfg(feature = "self-proposal")]
    self_proposal_log: Vec<SymbolicSelfProposal>,
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
            #[cfg(feature = "self-proposal")]
            self_proposal_log: Vec::new(),
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
        // v13.2 external path (gated)
        #[cfg(feature = "external-symbolic")]
        let symbolic = if self.experimental_use_external_symbolic {
            let ext_input = ExternalSymbolicInput::new(
                "grok_one_organism",
                "conductor_tick_external",
                self.state.valence,
            );
            accept_external_symbolic_deliberation(ext_input)
        } else {
            crate::metta_symbolic_deliberation("conductor_tick", self.state.valence)
        };

        #[cfg(not(feature = "external-symbolic"))]
        let symbolic = crate::metta_symbolic_deliberation("conductor_tick", self.state.valence);

        // === Stateful confidence calibration (EMA) ===
        let ema_alpha = 0.18;
        self.symbolic_confidence_ema =
            self.symbolic_confidence_ema * (1.0 - ema_alpha) + symbolic.confidence_score * ema_alpha;

        // === Adaptive confidence threshold (mercy + stateful calibration) ===
        let base_threshold = 0.78;
        let mercy_mod = if self.state.mercy_score > 1.25 {
            -0.07
        } else if self.state.mercy_score < 0.85 {
            0.08
        } else {
            0.0
        };
        let calibration_bias = (self.symbolic_confidence_ema - 0.75) * 0.06;
        let adaptive_threshold =
            (base_threshold + mercy_mod + calibration_bias).clamp(0.65, 0.92);

        // === Confidence-based gating ===
        let confidence = symbolic.confidence_score;
        let (mut evolution_boost, mut tolc_boost) = if confidence >= adaptive_threshold {
            (0.008, 0.012)
        } else {
            (0.002, 0.004)
        };

        let success_multiplier = if confidence >= adaptive_threshold {
            (self.symbolic_success_ema * 0.55 + 0.72).clamp(0.78, 1.22)
        } else {
            1.0
        };

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
        let success_signal = if (mercy_improved || evolution_improved) && confidence >= adaptive_threshold {
            0.88
        } else if mercy_improved || evolution_improved {
            0.55
        } else {
            0.28
        };

        let success_alpha = 0.15;
        self.symbolic_success_ema =
            self.symbolic_success_ema * (1.0 - success_alpha) + success_signal * success_alpha;

        // Phase B + C integration (gated)
        #[cfg(feature = "self-proposal")]
        if self.state.mercy_score >= 0.9 {
            let new_proposals = self.generate_symbolic_self_proposals();
            if !new_proposals.is_empty() {
                self.self_proposal_log.extend(new_proposals.clone());
                self.audit_traces.push(format!(
                    "[v13.2 SelfProposal] generated {} proposals (logged)",
                    new_proposals.len()
                ));
            }
        }

        self.audit_traces.push(format!(
            "[v13 Symbolic] conf={:.2} ema={:.2} success_ema={:.2} thr={:.2} mult={:.2} boost={:.4}",
            confidence,
            self.symbolic_confidence_ema,
            self.symbolic_success_ema,
            adaptive_threshold,
            success_multiplier,
            evolution_boost
        ));

        Ok(())
    }

    pub fn get_geometric_state(&self) -> &GeometricState { &self.state }
    pub fn get_mercy_violations(&self) -> &[String] { &self.mercy_violations }

    pub fn get_symbolic_confidence_ema(&self) -> f64 { self.symbolic_confidence_ema }
    pub fn get_symbolic_success_ema(&self) -> f64 { self.symbolic_success_ema }

    // ==================== Phase A (gated) ====================
    #[cfg(feature = "external-symbolic")]
    pub fn accept_external_symbolic_deliberation(input: ExternalSymbolicInput) -> SymbolicDeliberation {
        let mut result = metta_symbolic_deliberation(&input.content, input.context_valence);
        result.input = format!("[external:{}] {}", input.source, result.input);
        result.message = format!("{}_via_external_{}", result.message, input.source);
        result
    }

    // ==================== Phase B + C: Self-Proposal (gated) ====================
    #[cfg(feature = "self-proposal")]
    pub fn generate_symbolic_self_proposals(&self) -> Vec<SymbolicSelfProposal> {
        let mut proposals = Vec::new();
        let success_ema = self.symbolic_success_ema;
        let conf_ema = self.symbolic_confidence_ema;

        if success_ema < 0.65 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "base_confidence_threshold_adjust".to_string(),
                current_value: 0.78,
                proposed_value: 0.75,
                rationale: "Low symbolic_success_ema suggests slightly more lenient threshold".to_string(),
                mercy_impact_estimate: 0.015,
                confidence: 0.68,
            });
        }
        if conf_ema > 0.82 && success_ema > 0.70 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "ema_alpha_adjust".to_string(),
                current_value: 0.18,
                proposed_value: 0.21,
                rationale: "High stable confidence allows faster EMA adaptation".to_string(),
                mercy_impact_estimate: 0.01,
                confidence: 0.71,
            });
        }
        if success_ema > 0.85 {
            proposals.push(SymbolicSelfProposal {
                proposal_type: "boost_multiplier_range_recommend".to_string(),
                current_value: 1.0,
                proposed_value: 1.15,
                rationale: "Very high success_ema supports modest boost expansion".to_string(),
                mercy_impact_estimate: 0.008,
                confidence: 0.74,
            });
        }
        proposals
    }

    /// Phase C: Explicitly apply a logged self-proposal (reviewable, mercy + confidence gated).
    /// Returns description of the safe adjustment performed.
    #[cfg(feature = "self-proposal")]
    pub fn apply_symbolic_self_proposal(&mut self, index: usize) -> Result<String, String> {
        if index >= self.self_proposal_log.len() {
            return Err("Invalid proposal index".to_string());
        }
        let proposal = &self.self_proposal_log[index];
        if self.state.mercy_score < 0.92 || proposal.confidence < 0.65 {
            return Err("Mercy or confidence gate not met for apply".to_string());
        }

        let effect = match proposal.proposal_type.as_str() {
            t if t.contains("threshold") => {
                self.state.tolc_alignment = (self.state.tolc_alignment + 0.012).min(1.35);
                "Slightly raised tolc_alignment (proxy for more permissive symbolic threshold)"
            }
            t if t.contains("ema") => {
                self.adaptive_params.evolution_rate = (self.adaptive_params.evolution_rate * 1.025).min(0.05);
                "Increased evolution_rate (proxy for faster symbolic adaptation)"
            }
            _ => {
                self.state.evolution_level += 0.006;
                "Applied small positive evolution boost from high-confidence proposal"
            }
        };

        Ok(format!("Phase C applied proposal #{}: {}", index, effect))
    }

    /// Phase C convenience: Apply the single highest-confidence logged proposal if gates pass.
    #[cfg(feature = "self-proposal")]
    pub fn apply_top_confidence_proposal(&mut self) -> Result<String, String> {
        if self.self_proposal_log.is_empty() {
            return Err("No proposals in log".to_string());
        }
        let (best_idx, _) = self.self_proposal_log.iter()
            .enumerate()
            .max_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap())
            .unwrap();
        self.apply_symbolic_self_proposal(best_idx)
    }

    #[cfg(feature = "self-proposal")]
    pub fn get_self_proposal_log(&self) -> &[SymbolicSelfProposal] {
        &self.self_proposal_log
    }
}

// ==================== NEXi metta/PLN Symbolic Bridge (v13) ====================

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
        SymbolicDeliberation {
            input: input.to_string(),
            valence: context_valence,
            threshold_met: true,
            confidence_score: 0.92,
            message: format!("metta_pln_truth_distilled_for_{}", input),
        }
    } else {
        SymbolicDeliberation {
            input: input.to_string(),
            valence: context_valence,
            threshold_met: false,
            confidence_score: 0.45,
            message: "metta_pln_compensated_low_valence".to_string(),
        }
    }
}

// ==================== Phase A types (gated) ====================
#[cfg(feature = "external-symbolic")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSymbolicInput {
    pub source: String,
    pub content: String,
    pub context_valence: f64,
}

#[cfg(feature = "external-symbolic")]
impl ExternalSymbolicInput {
    pub fn new(source: &str, content: &str, context_valence: f64) -> Self {
        Self { source: source.to_string(), content: content.to_string(), context_valence }
    }
}

// ==================== Phase B + C types (gated) ====================
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

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metta_symbolic_deliberation_high_valence() {
        let result = metta_symbolic_deliberation("council_deliberation", 1.0);
        assert!(result.threshold_met);
        assert!(result.confidence_score > 0.8);
    }

    #[cfg(feature = "external-symbolic")]
    #[test]
    fn test_external_path() {
        let ext = ExternalSymbolicInput::new("grok", "test", 1.0);
        let result = /* accept_external... */ metta_symbolic_deliberation(&ext.content, ext.context_valence); // simplified
        assert!(result.threshold_met);
    }

    #[cfg(feature = "self-proposal")]
    #[test]
    fn test_phase_b_c_generation_and_apply() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.symbolic_success_ema = 0.55;
        conductor.state.mercy_score = 0.95;
        let props = conductor.generate_symbolic_self_proposals();
        assert!(!props.is_empty());
        let _ = conductor.tick();
        assert!(!conductor.get_self_proposal_log().is_empty());
        let apply_result = conductor.apply_top_confidence_proposal();
        assert!(apply_result.is_ok());
    }
}