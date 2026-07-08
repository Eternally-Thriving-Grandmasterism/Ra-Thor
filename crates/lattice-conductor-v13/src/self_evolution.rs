/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
/// 
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! SelfEvolutionOrchestrator — v13.4 Complete + v13.5 Planning
//!
//! v13.5 focus: PATSAGi Council Influence over Meta Rate Parameters.
//! Expanded tests + improved integration + council-specific weighting.
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
    meta_evolution_rate: f64,
    meta_audit_threshold: f64,
    meta_success_ema: f64,
    // v13.5: Council-specific influence weights
    council_weights: HashMap<String, f64>,
}

impl Default for SelfEvolutionOrchestrator {
    fn default() -> Self {
        let mut orch = Self {
            current_level: 0.0,
            total_evolutions: 0,
            blessings: HashMap::new(),
            evolution_history: Vec::new(),
            quantum_swarm_participation: 0.0,
            meta_evolution_rate: 0.01,
            meta_audit_threshold: 0.92,
            meta_success_ema: 0.70,
            council_weights: HashMap::new(),
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

    pub fn get_meta_evolution_rate(&self) -> f64 { self.meta_evolution_rate }
    pub fn get_meta_audit_threshold(&self) -> f64 { self.meta_audit_threshold }
    pub fn get_meta_success_ema(&self) -> f64 { self.meta_success_ema }

    pub fn stabilize_meta_rate(&mut self) {
        let base_rate: f64 = 0.01;
        let decay_rate: f64 = 0.08;
        self.meta_evolution_rate = self.meta_evolution_rate * (1.0 - decay_rate) + base_rate * decay_rate;
    }

    /// v13.5: Set or update influence weight for a specific council
    pub fn set_council_weight(&mut self, council_name: &str, weight: f64) {
        self.council_weights.insert(council_name.to_string(), weight.clamp(0.5, 2.0));
    }

    /// v13.5: Get influence weight for a council (defaults to 1.0)
    pub fn get_council_weight(&self, council_name: &str) -> f64 {
        *self.council_weights.get(council_name).unwrap_or(&1.0)
    }

    /// v13.5: Council meta rate adjust with expanded proposal types and strong mercy gating
    pub fn council_voted_meta_rate_adjust(
        &mut self,
        council_name: &str,
        proposal_type: &str,
        strength: f64,
        current_mercy: f64,
        trace_log: &mut Vec<String>,
    ) -> Result<String, String> {
        if current_mercy < 0.90 {
            return Err(format!("Council meta rate adjust blocked: insufficient mercy ({})", current_mercy));
        }
        if self.meta_success_ema < self.meta_audit_threshold {
            return Err(format!("Council meta rate adjust blocked: meta_success_ema too low"));
        }

        let weight = self.get_council_weight(council_name);
        let adjustment = strength * 0.015 * weight;

        let effect = match proposal_type {
            "increase_meta_rate" => {
                self.meta_evolution_rate = (self.meta_evolution_rate + adjustment).min(0.06);
                format!("increased meta_evolution_rate by {:.4} (weight={:.2})", adjustment, weight)
            }
            "decrease_meta_rate" => {
                self.meta_evolution_rate = (self.meta_evolution_rate - adjustment * 0.6).max(0.005);
                format!("decreased meta_evolution_rate by {:.4} (weight={:.2})", adjustment * 0.6, weight)
            }
            "tighten_meta_threshold" => {
                self.meta_audit_threshold = (self.meta_audit_threshold + adjustment * 0.5).min(0.96);
                format!("tightened meta_audit_threshold (weight={:.2})", weight)
            }
            "loosen_meta_threshold" => {
                self.meta_audit_threshold = (self.meta_audit_threshold - adjustment * 0.4).max(0.85);
                format!("loosened meta_audit_threshold (weight={:.2})", weight)
            }
            _ => "unknown proposal type".to_string(),
        };

        let event = format!("[v13.5 Council Meta] {} voted {} | mercy={:.2}", council_name, effect, current_mercy);
        self.evolution_history.push(event.clone());
        trace_log.push(event);

        Ok(effect)
    }

    /// Enhanced council_voted_evolution with meta rate influence (v13.5)
    pub fn council_voted_evolution(
        &mut self,
        council_name: &str,
        mercy_impact: f64,
        state: &mut GeometricState,
        trace_log: &mut Vec<String>,
    ) {
        let boost = (mercy_impact * 0.08).max(0.01);
        state.evolution_level += boost;
        state.mercy_score = (state.mercy_score + mercy_impact * 0.05).min(1.6);

        let event = format!("[Council-Voted Evolution] {} contributed {:.3} evolution boost", council_name, boost);
        self.evolution_history.push(event.clone());
        trace_log.push(event);

        self.current_level += boost * 0.4;
        self.total_evolutions += 1;

        // v13.5: High alignment triggers meta rate influence (weighted by council)
        if mercy_impact > 0.6 && state.mercy_score > 0.92 {
            let weight = self.get_council_weight(council_name);
            let strength = (mercy_impact * 0.6 * weight).min(1.0);
            let _ = self.council_voted_meta_rate_adjust(
                council_name,
                "increase_meta_rate",
                strength,
                state.mercy_score,
                trace_log,
            );
        }
    }

    /// Quantum Swarm integration hook
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
        self.stabilize_meta_rate();
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

        assert!(!meta_props.is_empty());
        assert!(meta_props.iter().any(|p| p.proposal_type == "meta_self_evolution_rate_increase"));
    }

    #[test]
    #[cfg(feature = "self-proposal")]
    fn test_generate_meta_proposals_from_conductor_returns_empty_when_conditions_not_met() {
        let mut conductor = SimpleLatticeConductor::new();
        conductor.state.mercy_score = 0.70;
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
        orchestrator.meta_success_ema = 0.95;

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

    #[test]
    fn test_v13_4_stabilize_meta_rate_prevents_runaway() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_evolution_rate = 0.04;

        for _ in 0..20 {
            orchestrator.stabilize_meta_rate();
        }

        assert!(orchestrator.get_meta_evolution_rate() < 0.025);
    }

    // ==================== v13.5 Tests ====================

    #[test]
    fn test_v13_5_council_weight_default_and_set() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        assert_eq!(orchestrator.get_council_weight("TestCouncil"), 1.0);

        orchestrator.set_council_weight("TestCouncil", 1.5);
        assert_eq!(orchestrator.get_council_weight("TestCouncil"), 1.5);
    }

    #[test]
    #[cfg(feature = "self-proposal")]
    fn test_v13_5_council_meta_rate_adjust_blocked_by_low_mercy() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        let result = orchestrator.council_voted_meta_rate_adjust(
            "TestCouncil",
            "increase_meta_rate",
            0.8,
            0.75, // too low
            &mut vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "self-proposal")]
    fn test_v13_5_council_meta_rate_adjust_with_weight() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_success_ema = 0.95;
        orchestrator.set_council_weight("PowerfulCouncil", 1.8);

        let result = orchestrator.council_voted_meta_rate_adjust(
            "PowerfulCouncil",
            "increase_meta_rate",
            0.8,
            0.95,
            &mut vec![],
        );

        assert!(result.is_ok());
        let msg = result.unwrap();
        assert!(msg.contains("weight=1.8"));
    }

    #[test]
    fn test_v13_5_council_voted_evolution_triggers_meta_when_aligned() {
        let mut orchestrator = SelfEvolutionOrchestrator::new();
        orchestrator.meta_success_ema = 0.95;
        let mut state = GeometricState { mercy_score: 0.95, ..Default::default() };
        let mut trace = vec![];

        orchestrator.council_voted_evolution("AlignedCouncil", 0.8, &mut state, &mut trace);

        // Should have triggered meta rate adjustment
        assert!(orchestrator.get_meta_evolution_rate() > 0.01);
    }
}
