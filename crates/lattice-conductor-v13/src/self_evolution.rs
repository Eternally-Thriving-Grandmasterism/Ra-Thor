/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
/// 
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

//! SelfEvolutionOrchestrator — v13.5 + Start of Idea 2 (Light Meta-Strategy Archive)
//!
//! Foundation for light meta-strategy profiles.

use crate::{GeometricState, SimpleLatticeConductor, ConductorSymbolicParameters};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ==================== v13.5: Meta Strategy Profiles (Idea 2 Foundation) ====================

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
    // v13.5 Idea 2: Current meta strategy profile
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

    // ... existing methods ...

    // ==================== v13.5 Idea 2: Light Meta-Strategy Profile ====================

    pub fn get_current_meta_profile(&self) -> MetaStrategyProfile {
        self.current_meta_profile
    }

    pub fn set_meta_profile(&mut self, profile: MetaStrategyProfile) {
        self.current_meta_profile = profile;
        let event = format!("[v13.5 MetaProfile] Switched to {:?} - {}", profile, profile.description());
        self.evolution_history.push(event);
    }

    /// Council can propose switching meta strategy profile (foundation for Idea 2)
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

        let old = self.current_meta_profile;
        self.set_meta_profile(new_profile);

        let event = format!(
            "[v13.5 Council MetaProfile] {} proposed switch {:?} → {:?}",
            council_name, old, new_profile
        );
        self.evolution_history.push(event.clone());
        trace_log.push(event.clone());

        Ok(event)
    }

    // ... rest of existing methods (council_voted_meta_rate_adjust, critique, etc.) ...
}
