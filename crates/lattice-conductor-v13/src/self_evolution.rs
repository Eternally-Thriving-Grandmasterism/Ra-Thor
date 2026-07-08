/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
/// 
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! SelfEvolutionOrchestrator — v13.5 Idea 2: Full Profile Switching + Integration
//!
//! MetaStrategyProfile now meaningfully affects meta parameters.

use crate::{GeometricState, SimpleLatticeConductor, ConductorSymbolicParameters};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaStrategyProfile {
    Conservative,
    Balanced,
    Aggressive,
}

impl Default for MetaStrategyProfile {
    fn default() -> Self { MetaStrategyProfile::Balanced }
}

impl MetaStrategyProfile {
    pub fn description(&self) -> &'static str {
        match self {
            MetaStrategyProfile::Conservative => "Lower meta rate, stricter thresholds, high stability focus",
            MetaStrategyProfile::Balanced => "Moderate meta rate and thresholds, balanced evolution",
            MetaStrategyProfile::Aggressive => "Higher meta rate, more permissive thresholds, faster adaptation",
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
    council_weights: HashMap<String, f64>,
    decision_type_multipliers: HashMap<(String, MetaDecisionType), f64>,
    current_meta_profile: MetaStrategyProfile,
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
            current_meta_profile: MetaStrategyProfile::default(),
        };
        orch.register_default_blessings();
        orch
    }
}

impl SelfEvolutionOrchestrator {
    pub fn new() -> Self { Self::default() }

    fn register_default_blessings(&mut self) { /* ... */ }

    pub fn get_meta_evolution_rate(&self) -> f64 { self.meta_evolution_rate }
    pub fn get_meta_audit_threshold(&self) -> f64 { self.meta_audit_threshold }
    pub fn get_meta_success_ema(&self) -> f64 { self.meta_success_ema }

    pub fn stabilize_meta_rate(&mut self) {
        let base_rate: f64 = 0.01;
        let decay_rate: f64 = match self.current_meta_profile {
            MetaStrategyProfile::Conservative => 0.12,
            MetaStrategyProfile::Balanced => 0.08,
            MetaStrategyProfile::Aggressive => 0.05,
        };
        self.meta_evolution_rate = self.meta_evolution_rate * (1.0 - decay_rate) + base_rate * decay_rate;
    }

    // ==================== v13.5 Council Weighting ====================

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

    // ==================== Profile-Aware Meta Parameters ====================

    /// Applies the effects of the current meta profile to the core parameters.
    fn apply_profile_effects(&mut self) {
        match self.current_meta_profile {
            MetaStrategyProfile::Conservative => {
                self.meta_evolution_rate = self.meta_evolution_rate.min(0.018);
                self.meta_audit_threshold = self.meta_audit_threshold.max(0.935);
            }
            MetaStrategyProfile::Balanced => {
                // No strong bias, keep current values within reasonable bounds
                self.meta_evolution_rate = self.meta_evolution_rate.clamp(0.008, 0.025);
                self.meta_audit_threshold = self.meta_audit_threshold.clamp(0.90, 0.94);
            }
            MetaStrategyProfile::Aggressive => {
                self.meta_evolution_rate = self.meta_evolution_rate.max(0.022);
                self.meta_audit_threshold = self.meta_audit_threshold.min(0.91);
            }
        }
    }

    pub fn set_meta_profile(&mut self, profile: MetaStrategyProfile) {
        let old = self.current_meta_profile;
        self.current_meta_profile = profile;
        self.apply_profile_effects();

        let event = format!(
            "[v13.5 MetaProfile] Switched {:?} → {:?} | rate={:.3} thr={:.3}",
            old, profile, self.meta_evolution_rate, self.meta_audit_threshold
        );
        self.evolution_history.push(event);
    }

    pub fn get_current_meta_profile(&self) -> MetaStrategyProfile {
        self.current_meta_profile
    }

    // ==================== Council Profile Switching ====================

    pub fn council_propose_meta_profile_switch(
        &mut self,
        council_name: &str,
        new_profile: MetaStrategyProfile,
        current_mercy: f64,
        trace_log: &mut Vec<String>,
    ) -> Result<String, String> {
        if current_mercy < 0.93 {
            return Err("Insufficient mercy to propose meta profile switch".to_string());
        }

        self.set_meta_profile(new_profile);

        let event = format!(
            "[v13.5 Council MetaProfile] {} proposed switch to {:?}",
            council_name, new_profile
        );
        self.evolution_history.push(event.clone());
        trace_log.push(event.clone());

        Ok(event)
    }

    // ==================== Refined Critique (profile aware) ====================

    pub fn perform_meta_critique(
        &self,
        council_name: &str,
        proposal_type: &str,
        requested_strength: f64,
        current_mercy: f64,
    ) -> (bool, String) {
        let mut concerns = Vec::new();
        let mut passed = true;

        if current_mercy < 0.90 {
            concerns.push("Low current mercy".to_string());
            passed = false;
        }

        if self.meta_success_ema < 0.75 {
            concerns.push(format!("Low meta_success_ema ({:.2})", self.meta_success_ema));
            passed = false;
        }

        let is_rate_increase = proposal_type.contains("increase_meta_rate");
        if is_rate_increase && self.meta_evolution_rate > 0.037 && requested_strength > 0.55 {
            concerns.push(format!("Meta rate already elevated ({:.3})", self.meta_evolution_rate));
            passed = false;
        }

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

        let is_threshold_change = proposal_type.contains("threshold");
        if is_threshold_change && requested_strength > 0.70 && current_mercy < 0.935 {
            concerns.push("High-strength threshold change at moderate mercy".to_string());
            passed = false;
        }

        // Profile-aware: Aggressive profile is more tolerant of change
        if self.current_meta_profile == MetaStrategyProfile::Aggressive && !passed && current_mercy > 0.92 {
            passed = true; // relax one concern in Aggressive mode
        }

        let critique_result = if passed {
            format!("Critique passed for {} (strength={:.2})", council_name, requested_strength)
        } else {
            format!("Critique concerns: {}", concerns.join("; "))
        };

        (passed, critique_result)
    }

    // ... rest of the methods (council_voted_meta_rate_adjust, etc.) remain ...
}
