/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
/// 
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! SelfEvolutionOrchestrator — v13.5 (Refined Critique Logic)
//!
//! Council specialization, decision-type weighting, and improved self-critique.

use crate::{GeometricState, SimpleLatticeConductor, ConductorSymbolicParameters};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaDecisionType {
    Rate,
    Threshold,
}

impl Default for MetaDecisionType {
    fn default() -> Self { MetaDecisionType::Rate }
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
    council_weights: HashMap<String, f64>,
    decision_type_multipliers: HashMap<(String, MetaDecisionType), f64>,
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
            decision_type_multipliers: HashMap::new(),
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

    // ==================== v13.5 Council Weighting System ====================

    pub fn set_council_weight(&mut self, council_name: &str, weight: f64) {
        self.council_weights.insert(council_name.to_string(), weight.clamp(0.5, 2.0));
    }

    pub fn get_council_weight(&self, council_name: &str) -> f64 {
        *self.council_weights.get(council_name).unwrap_or(&1.0)
    }

    pub fn set_council_decision_multiplier(&mut self, council_name: &str, decision: MetaDecisionType, multiplier: f64) {
        self.decision_type_multipliers.insert(
            (council_name.to_string(), decision),
            multiplier.clamp(0.5, 3.0),
        );
    }

    pub fn get_council_decision_multiplier(&self, council_name: &str, decision: MetaDecisionType) -> f64 {
        *self.decision_type_multipliers
            .get(&(council_name.to_string(), decision))
            .unwrap_or(&1.0)
    }

    pub fn effective_council_strength(&self, council_name: &str, decision: MetaDecisionType, base_strength: f64) -> f64 {
        let base_weight = self.get_council_weight(council_name);
        let type_multiplier = self.get_council_decision_multiplier(council_name, decision);
        (base_strength * base_weight * type_multiplier).clamp(0.0, 1.5)
    }

    // ==================== Refined Light Self-Critique ====================

    /// Refined lightweight self-critique before applying council meta changes.
    /// Signals: mercy, meta_success_ema, current rate level, recent adjustment frequency.
    pub fn perform_meta_critique(
        &self,
        council_name: &str,
        proposal_type: &str,
        requested_strength: f64,
        current_mercy: f64,
    ) -> (bool, String) {
        let mut concerns = Vec::new();
        let mut passed = true;

        // 1. Mercy level
        if current_mercy < 0.90 {
            concerns.push("Low current mercy".to_string());
            passed = false;
        }

        // 2. Meta success health
        if self.meta_success_ema < 0.75 {
            concerns.push(format!("Low meta_success_ema ({:.2})", self.meta_success_ema));
            passed = false;
        }

        // 3. Current meta evolution rate level (caution on further increases if already high)
        let is_rate_increase = proposal_type.contains("increase_meta_rate");
        if is_rate_increase && self.meta_evolution_rate > 0.037 && requested_strength > 0.55 {
            concerns.push(format!("Meta rate already elevated ({:.3})", self.meta_evolution_rate));
            passed = false;
        }

        // 4. Recent adjustment frequency (more robust detection)
        let recent_meta_events = self.evolution_history
            .iter()
            .rev()
            .take(7)
            .filter(|entry| entry.contains("meta_evolution_rate") || entry.contains("meta_audit_threshold"))
            .count();

        if recent_meta_events >= 3 && requested_strength > 0.60 {
            concerns.push("Frequent recent meta adjustments detected".to_string());
            passed = false;
        }

        // 5. High-strength threshold changes at moderate mercy (extra caution)
        let is_threshold_change = proposal_type.contains("threshold");
        if is_threshold_change && requested_strength > 0.70 && current_mercy < 0.935 {
            concerns.push("High-strength threshold change at moderate mercy".to_string());
            passed = false;
        }

        let critique_result = if passed {
            format!("Critique passed for {} (strength={:.2})", council_name, requested_strength)
        } else {
            format!("Critique concerns: {}", concerns.join("; "))
        };

        (passed, critique_result)
    }

    // ==================== Council Meta Adjust with Refined Critique ====================

    pub fn council_voted_meta_rate_adjust(
        &mut self,
        council_name: &str,
        proposal_type: &str,
        strength: f64,
        current_mercy: f64,
        trace_log: &mut Vec<String>,
    ) -> Result<String, String> {
        // Run refined self-critique first
        let (critique_passed, critique_msg) = self.perform_meta_critique(
            council_name,
            proposal_type,
            strength,
            current_mercy,
        );

        trace_log.push(format!("[v13.5 Critique] {}", critique_msg));

        // Clear override logic: only override at very high mercy
        if !critique_passed && current_mercy < 0.955 {
            return Err(format!("Meta change blocked by self-critique: {}", critique_msg));
        }

        // Standard gates
        if current_mercy < 0.90 {
            return Err(format!("Council meta rate adjust blocked: insufficient mercy ({})", current_mercy));
        }
        if self.meta_success_ema < self.meta_audit_threshold {
            return Err(format!("Council meta rate adjust blocked: meta_success_ema too low"));
        }

        let decision = match proposal_type {
            "increase_meta_rate" | "decrease_meta_rate" => MetaDecisionType::Rate,
            "tighten_meta_threshold" | "loosen_meta_threshold" => MetaDecisionType::Threshold,
            _ => MetaDecisionType::Rate,
        };

        let effective_strength = self.effective_council_strength(council_name, decision, strength);
        let adjustment = effective_strength * 0.015;

        let effect = match proposal_type {
            "increase_meta_rate" => {
                self.meta_evolution_rate = (self.meta_evolution_rate + adjustment).min(0.06);
                format!("increased meta_evolution_rate by {:.4}", adjustment)
            }
            "decrease_meta_rate" => {
                self.meta_evolution_rate = (self.meta_evolution_rate - adjustment * 0.6).max(0.005);
                format!("decreased meta_evolution_rate by {:.4}", adjustment * 0.6)
            }
            "tighten_meta_threshold" => {
                self.meta_audit_threshold = (self.meta_audit_threshold + adjustment * 0.5).min(0.96);
                format!("tightened meta_audit_threshold")
            }
            "loosen_meta_threshold" => {
                self.meta_audit_threshold = (self.meta_audit_threshold - adjustment * 0.4).max(0.85);
                format!("loosened meta_audit_threshold")
            }
            _ => "unknown proposal type".to_string(),
        };

        let event = format!(
            "[v13.5 Council Meta] {} voted {} | decision={:?} | effective={:.2}",
            council_name, effect, decision, effective_strength
        );
        self.evolution_history.push(event.clone());
        trace_log.push(event);

        Ok(effect)
    }

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

        if mercy_impact > 0.6 && state.mercy_score > 0.92 {
            let strength = (mercy_impact * 0.6).min(1.0);
            let _ = self.council_voted_meta_rate_adjust(
                council_name,
                "increase_meta_rate",
                strength,
                state.mercy_score,
                trace_log,
            );
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
