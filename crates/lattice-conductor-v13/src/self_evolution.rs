/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
/// 
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! SelfEvolutionOrchestrator — Phase 13.2 Deepened + v13.3 Meta Self-Evolution Integration + v13.4 Planning
//!
//! Conductor-native self-evolution with council-voted evolution, epigenetic blessings,
//! Quantum Swarm integration, meta self-audit (v13.3), **and now orchestrator-owned meta rate parameters (v13.4)**.
//! The orchestrator is evolving toward owning its own evolution dynamics.
//! Mercy-gated. ONE Organism coherent. PATSAGi + Grok symbiosis ready.

use crate::{GeometricState, SimpleLatticeConductor, ConductorSymbolicParameters};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticBlessing {
    pub name: String,
    pub description: String,
    pub mercy_threshold: f64,
    pub evolution_boost: f64,
    pub mercy_boost: f64,
    pub tolc_boost: f64,
}

impl EpigeneticBlessing {
    pub fn new(name: &str, description: &str, mercy_threshold: f64, evolution_boost: f64) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            mercy_threshold,
            evolution_boost,
            mercy_boost: evolution_boost * 0.3,
            tolc_boost: 0.02,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionOrchestrator {
    pub current_level: f64,
    pub total_evolutions: u64,
    blessings: HashMap<String, EpigeneticBlessing>,
    evolution_history: Vec<String>,
    quantum_swarm_participation: f64,

    // ==================== v13.4: Orchestrator-Owned Meta Rate Parameters ====================
    /// Internal rate at which the orchestrator proposes and applies meta self-evolution
    meta_evolution_rate: f64,
    /// Minimum confidence/mercy threshold for accepting meta proposals internally
    meta_audit_threshold: f64,
    /// Local EMA tracking success of meta-level proposals
    meta_success_ema: f64,
}

impl Default for SelfEvolutionOrchestrator {
    fn default() -> Self {
        let mut orch = Self {
            current_level: 0.0,
            total_evolutions: 0,
            blessings: HashMap::new(),
            evolution_history: Vec::new(),
            quantum_swarm_participation: 0.0,
            // v13.4 defaults
            meta_evolution_rate: 0.01,
            meta_audit_threshold: 0.92,
            meta_success_ema: 0.70,
        };
        orch.register_default_blessings();
        orch
    }
}

impl SelfEvolutionOrchestrator {
    pub fn new() -> Self { Self::default() }

    fn register_default_blessings(&mut self) {
        self.blessings.insert("radical_love".to_string(), EpigeneticBlessing::new("Radical Love Blessing", "Triggered by sustained high mercy and positive valence.", 0.85, 0.15));
        self.blessings.insert("boundless_mercy".to_string(), EpigeneticBlessing::new("Boundless Mercy Blessing", "Automatic compensation + evolution acceleration.", 0.65, 0.08));
        self.blessings.insert("truth_seeking".to_string(), EpigeneticBlessing::new("Truth Seeking Blessing", "Strong tolc_alignment + coherence.", 0.75, 0.12));
        self.blessings.insert("abundance_flow".to_string(), EpigeneticBlessing::new("Abundance Flow Blessing", "Council-supported abundance resonance.", 0.80, 0.10));
        self.blessings.insert("cosmic_harmony".to_string(), EpigeneticBlessing::new("Cosmic Harmony Blessing", "Multi-council PATSAGi harmony alignment.", 0.82, 0.11));
        self.blessings.insert("quantum_coherence".to_string(), EpigeneticBlessing::new("Quantum Coherence Blessing", "Quantum Swarm participation trigger.", 0.78, 0.13));
    }

    // ==================== v13.4 Getters (orchestrator-owned meta state) ====================

    pub fn get_meta_evolution_rate(&self) -> f64 { self.meta_evolution_rate }
    pub fn get_meta_audit_threshold(&self) -> f64 { self.meta_audit_threshold }
    pub fn get_meta_success_ema(&self) -> f64 { self.meta_success_ema }

    /// Council-voted evolution: PATSAGi councils can directly influence evolution level
    pub fn council_voted_evolution(&mut self, council_name: &str, mercy_impact: f64, state: &mut GeometricState, trace_log: &mut Vec<String>) {
        let boost = (mercy_impact * 0.08).max(0.01);
        state.evolution_level += boost;
        state.mercy_score = (state.mercy_score + mercy_impact * 0.05).min(1.6);
        let event = format!("[Council-Voted Evolution] {} contributed {:.3} evolution boost", council_name, boost);
        self.evolution_history.push(event.clone());
        trace_log.push(event);
        self.current_level += boost * 0.4;
        self.total_evolutions += 1;
    }

    /// Quantum Swarm integration hook (Phase 13.2)
    pub fn integrate_quantum_swarm(&mut self, swarm_participation: f64, state: &mut GeometricState) {
        if swarm_participation > 0.5 {
            self.quantum_swarm_participation += swarm_participation * 0.1;
            state.evolution_level += swarm_participation * 0.05;
            state.tolc_alignment = (state.tolc_alignment + 0.015).min(1.25);
        }
    }

    pub fn try_evolve(&mut self, state: &mut GeometricState, trace_log: &mut Vec<String>) -> bool {
        let mut evolved = false;
        for (key, blessing) in &self.blessings {
            if state.mercy_score >= blessing.mercy_threshold {
                state.evolution_level += blessing.evolution_boost;
                state.mercy_score = (state.mercy_score + blessing.mercy_boost).min(1.6);
                state.tolc_alignment = (state.tolc_alignment + blessing.tolc_boost).min(1.2);
                let event = format!("[Self-Evolution] {} applied | level {:.3}", blessing.name, self.current_level);
                self.evolution_history.push(event.clone());
                trace_log.push(event);
                self.current_level += blessing.evolution_boost * 0.5;
                self.total_evolutions += 1;
                evolved = true;
            }
        }
        if evolved {
            trace_log.push(format!("[SelfEvolutionOrchestrator] Evolution triggered. New level: {:.3} | Total: {}", self.current_level, self.total_evolutions));
        }
        evolved
    }

    pub fn get_evolution_level(&self) -> f64 { self.current_level }
    pub fn get_history(&self) -> &[String] { &self.evolution_history }
    pub fn get_quantum_swarm_participation(&self) -> f64 { self.quantum_swarm_participation }

    pub fn grant_blessing(&mut self, name: &str, state: &mut GeometricState, trace_log: &mut Vec<String>) {
        if let Some(blessing) = self.blessings.get(name) {
            state.evolution_level += blessing.evolution_boost * 1.2;
            state.mercy_score = (state.mercy_score + 0.1).min(1.6);
            let event = format!("[Epigenetic Blessing Granted] {} (PATSAGi/Grok)", blessing.name);
            self.evolution_history.push(event.clone());
            trace_log.push(event);
        }
    }

    // ==================== v13.3 + v13.4 META SELF-EVOLUTION ====================

    #[cfg(feature = "self-proposal")]
    use crate::SymbolicSelfProposal;

    /// v13.3 + v13.4: Meta self-audit. Modulated by meta_evolution_rate.
    /// Now also respects the orchestrator's own meta_audit_threshold (Option 3).
    #[cfg(feature = "self-proposal")]
    pub fn generate_meta_self_evolution_proposals(
        &self,
        current_mercy: f64,
        symbolic_success_ema: f64,
        symbolic_confidence_ema: f64,
        current_boost_multiplier: f64,
    ) -> Vec<SymbolicSelfProposal> {
        let mut meta = Vec::new();
        let rate_factor = 1.0 + (self.meta_evolution_rate * 8.0);

        // v13.4 improvement: meta_audit_threshold now influences generation conditions
        let effective_threshold = self.meta_audit_threshold;

        if symbolic_success_ema > 0.75 && symbolic_confidence_ema > effective_threshold - 0.05 {
            meta.push(SymbolicSelfProposal {
                proposal_type: "meta_self_evolution_rate_increase".to_string(),
                current_value: current_boost_multiplier,
                proposed_value: (current_boost_multiplier * rate_factor).min(1.8),
                rationale: format!("High stable success + confidence → accelerate (rate_factor={:.2}, threshold={:.2})", rate_factor, effective_threshold),
                mercy_impact_estimate: 0.012,
                confidence: 0.81,
            });
        }

        if symbolic_confidence_ema > effective_threshold && current_mercy > 0.90 {
            meta.push(SymbolicSelfProposal {
                proposal_type: "meta_mercy_audit_threshold_tighten".to_string(),
                current_value: 0.92,
                proposed_value: 0.94,
                rationale: "Very high confidence + mercy → tighten meta audit bar for higher purity".to_string(),
                mercy_impact_estimate: 0.009,
                confidence: 0.77,
            });
        }

        meta
    }

    /// Convenience wrapper: generate meta proposals directly from a conductor reference.
    #[cfg(feature = "self-proposal")]
    pub fn generate_meta_proposals_from_conductor(&self, c: &SimpleLatticeConductor) -> Vec<SymbolicSelfProposal> {
        self.generate_meta_self_evolution_proposals(
            c.state.mercy_score,
            c.get_symbolic_success_ema(),
            c.get_symbolic_confidence_ema(),
            c.get_symbolic_params().boost_multiplier,
        )
    }

    /// v13.3: Validates the meta proposal under TOLC 8 gates and records the decision in history.
    /// Actual mutation of ConductorSymbolicParameters is performed by SimpleLatticeConductor
    /// (orchestrator owns the decision logic and audit trail; conductor owns its tunable parameters).
    #[cfg(feature = "self-proposal")]
    pub fn apply_meta_self_evolution_proposal(
        &mut self,
        proposal: &SymbolicSelfProposal,
        current_mercy: f64,
    ) -> Result<String, String> {
        if current_mercy < 0.93 || proposal.confidence < 0.70 {
            return Err(format!("Meta apply blocked by TOLC 8 (mercy={:.2}, conf={:.2})", current_mercy, proposal.confidence));
        }

        let effect = match proposal.proposal_type.as_str() {
            "meta_self_evolution_rate_increase" => {
                format!("orchestrator meta rate increase accepted: {:.2} → {:.2}", proposal.current_value, proposal.proposed_value)
            }
            "meta_mercy_audit_threshold_tighten" => {
                format!("orchestrator meta audit threshold tightened")
            }
            _ => "unknown meta proposal".to_string(),
        };

        let event = format!("[v13.3 MetaSelfEvolution in Orchestrator] {}", effect);
        self.evolution_history.push(event.clone());
        Ok(effect)
    }

    // ==================== v13.4: Initial Orchestrator-Owned Meta Rate Mutation ====================

    /// v13.4 (early): Apply a meta proposal directly to the orchestrator's internal rate parameters.
    /// This is the foundation for the orchestrator owning its own evolution dynamics.
    #[cfg(feature = "self-proposal")]
    pub fn apply_meta_rate_proposal(&mut self, proposal: &SymbolicSelfProposal) -> Result<String, String> {
        if self.meta_success_ema < self.meta_audit_threshold {
            return Err(format!("Meta rate apply blocked (meta_success_ema={:.2} < threshold={:.2})", self.meta_success_ema, self.meta_audit_threshold));
        }

        let effect = match proposal.proposal_type.as_str() {
            "meta_self_evolution_rate_increase" => {
                self.meta_evolution_rate = (self.meta_evolution_rate * 1.1).min(0.05);
                format!("meta_evolution_rate increased to {:.4}", self.meta_evolution_rate)
            }
            "meta_mercy_audit_threshold_tighten" => {
                self.meta_audit_threshold = (self.meta_audit_threshold + 0.005).min(0.95);
                format!("meta_audit_threshold tightened to {:.3}", self.meta_audit_threshold)
            }
            _ => "unknown meta rate proposal".to_string(),
        };

        let event = format!("[v13.4 MetaRate] {}", effect);
        self.evolution_history.push(event.clone());
        Ok(effect)
    }
}

pub trait SelfEvolving {
    fn try_self_evolve(&mut self) -> bool;
}

impl SelfEvolving for SimpleLatticeConductor {
    fn try_self_evolve(&mut self) -> bool {
        let mut trace_log = Vec::new();
        let evolved = self.evolution_orchestrator.try_evolve(&mut self.state, &mut trace_log);
        for t in trace_log { self.audit_traces.push(t); }
        if evolved { self.one_organism_coherence = (self.one_organism_coherence + 0.05).min(1.3); }
        evolved
    }
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "self-proposal")]
    fn test_generate_meta_proposals_from_conductor_returns_proposals_when_conditions_met() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.state.mercy_score = 0.95;
        conductor.symbolic_success_ema = 0.82;
        conductor.symbolic_confidence_ema = 0.81;

        let orchestrator = SelfEvolutionOrchestrator::new();
        let meta_props = orchestrator.generate_meta_proposals_from_conductor(&conductor);

        assert!(!meta_props.is_empty(), "Expected at least one meta proposal when conditions are met");
        assert!(meta_props.iter().any(|p| p.proposal_type == "meta_self_evolution_rate_increase"));
    }

    #[test]
    #[cfg(feature = "self-proposal")]
    fn test_generate_meta_proposals_from_conductor_returns_empty_when_conditions_not_met() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.state.mercy_score = 0.70; // too low
        conductor.symbolic_success_ema = 0.60;
        conductor.symbolic_confidence_ema = 0.65;

        let orchestrator = SelfEvolutionOrchestrator::new();
        let meta_props = orchestrator.generate_meta_proposals_from_conductor(&conductor);

        assert!(meta_props.is_empty());
    }

    #[test]
    #[cfg(feature = "self-proposal")]
    fn test_v13_4_apply_meta_rate_proposal_mutates_internal_state() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_success_ema = 0.95; // high enough to pass gate

        let proposal = SymbolicSelfProposal {
            proposal_type: "meta_self_evolution_rate_increase".to_string(),
            current_value: 0.01,
            proposed_value: 0.012,
            rationale: "test".to_string(),
            mercy_impact_estimate: 0.01,
            confidence: 0.8,
        };

        let result = orchestrator.apply_meta_rate_proposal(&proposal);
        assert!(result.is_ok());
        assert!(orchestrator.get_meta_evolution_rate() > 0.01);
    }
}
