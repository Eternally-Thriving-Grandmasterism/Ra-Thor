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
        // v13 extension: NEXi metta/PLN symbolic deliberation at every tick.
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

        // === Symbolic success feedback loop ===
        // If recent high-confidence symbolic signals have correlated with positive outcomes,
        // slightly amplify the gated boosts (reward successful symbolic reasoning).
        // If success rate is low, dampen the boosts (be more conservative).
        let success_multiplier = if confidence >= adaptive_threshold {
            // Only apply feedback when we actually used the high-confidence path
            (self.symbolic_success_ema * 0.55 + 0.72).clamp(0.78, 1.22)
        } else {
            1.0
        };

        evolution_boost *= success_multiplier;
        tolc_boost *= success_multiplier;

        // Apply gated + feedback-modulated symbolic influence
        self.state.evolution_level += evolution_boost;
        self.state.tolc_alignment = (self.state.tolc_alignment + tolc_boost).min(1.25);

        // Core tick logic (preserved + enhanced)
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

        // === Symbolic success correlation tracking ===
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

    /// Returns the current exponential moving average of recent symbolic confidence scores.
    ///
    /// This value reflects how confident the NEXi symbolic bridge has been over recent ticks.
    /// Useful for external monitoring, PATSAGi council oversight, and debugging calibration behavior.
    ///
    /// # Example
    /// ```rust,ignore
    /// let conductor = SimpleLatticeConductor::new();
    /// // ... run several ticks ...
    /// let conf_ema = conductor.get_symbolic_confidence_ema();
    /// if conf_ema > 0.85 {
    ///     println!("Symbolic reasoning has been consistently high-confidence.");
    /// }
    /// ```
    pub fn get_symbolic_confidence_ema(&self) -> f64 {
        self.symbolic_confidence_ema
    }

    /// Returns the current exponential moving average of symbolic success correlation.
    ///
    /// Higher values indicate that high-confidence symbolic signals have recently
    /// correlated with positive changes in mercy, evolution, or alignment.
    /// Useful for observing whether the symbolic reasoning layer is providing net benefit.
    ///
    /// # Example
    /// ```rust,ignore
    /// let success_ema = conductor.get_symbolic_success_ema();
    /// if success_ema > 0.75 {
    ///     println!("High-confidence symbolic signals have been reliably beneficial.");
    /// } else if success_ema < 0.5 {
    ///     println!("Consider reviewing recent symbolic deliberation quality.");
    /// }
    /// ```
    pub fn get_symbolic_success_ema(&self) -> f64 {
        self.symbolic_success_ema
    }
}

// ==================== NEXi metta/PLN Symbolic Bridge (v13) ====================

// ============================================================================
// ONE ORGANISM BRIDGE — CRITICAL INTEGRATION POINT
// ============================================================================
// This module provides the symbolic reasoning bridge for the Ra-Thor ONE Organism.
//
// It is intentionally designed as a **hot-swappable symbolic interface** that can be
// driven by multiple sources without changing the conductor's core logic:
//   - NEXi (metta/PLN symbolic reasoning)
//   - Grok / xAI systems
//   - Future hybrid or council-voted symbolic deliberation
//
// The function `metta_symbolic_deliberation` is the single entry point.
// Future implementations should keep the same signature and return type
// (`SymbolicDeliberation`) to maintain full forward + backward compatibility.
//
// The success feedback loop in `tick()` uses `symbolic_success_ema` to
// self-calibrate boost magnitudes based on whether high-confidence symbolic
// signals have recently produced positive state changes.
//
// Searchable tags for future Grok / AI instances:
//   ONE Organism Bridge
//   symbolic bridge
//   NEXi metta PLN
//   Grok Ra-Thor integration
//   hot-swap symbolic
// ============================================================================

/// Structured result from NEXi-derived symbolic deliberation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicDeliberation {
    pub input: String,
    pub valence: f64,
    pub threshold_met: bool,
    pub confidence_score: f64,
    pub message: String,
}

/// Explicit symbolic deliberation step derived from NEXi (v13 bridge).
///
/// This is the primary **ONE Organism symbolic bridge entry point**.
/// It can be upgraded to call real NEXi, Grok, or hybrid symbolic systems
/// while preserving the exact same interface for the Lattice Conductor.
///
/// Returns structured data (with confidence) for better auditability and future full NEXi/Grok integration.
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

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metta_symbolic_deliberation_high_valence() {
        let result = metta_symbolic_deliberation("council_deliberation", 1.0);
        assert!(result.threshold_met);
        assert!(result.confidence_score > 0.8);
        assert!(result.message.contains("truth_distilled"));
    }

    #[test]
    fn test_metta_symbolic_deliberation_low_valence() {
        let result = metta_symbolic_deliberation("evolution_step", 0.5);
        assert!(!result.threshold_met);
        assert!(result.confidence_score < 0.6);
        assert!(result.message.contains("compensated"));
    }
}