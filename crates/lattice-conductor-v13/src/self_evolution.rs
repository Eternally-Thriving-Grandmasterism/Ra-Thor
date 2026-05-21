//! SelfEvolutionOrchestrator — Phase 13.2
//! 
//! Conductor-native self-evolution system with epigenetic blessing hooks.
//! Mercy-gated evolution. ONE Organism coherent. Sovereign and auditable.
//!
//! This realizes the self-evolution as conductor-native direction from the v13 roadmap.

use crate::{GeometricState, SimpleLatticeConductor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a single epigenetic blessing that can be applied during evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticBlessing {
    pub name: String,
    pub description: String,
    pub mercy_threshold: f64,      // Minimum mercy_score required to receive this blessing
    pub evolution_boost: f64,      // How much it increases evolution_level
    pub mercy_boost: f64,          // Secondary mercy reinforcement
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

/// The orchestrator that manages and applies self-evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionOrchestrator {
    pub current_level: f64,
    pub total_evolutions: u64,
    blessings: HashMap<String, EpigeneticBlessing>,
    evolution_history: Vec<String>, // Auditable trace of evolution events
}

impl Default for SelfEvolutionOrchestrator {
    fn default() -> Self {
        let mut orch = Self {
            current_level: 0.0,
            total_evolutions: 0,
            blessings: HashMap::new(),
            evolution_history: Vec::new(),
        };
        orch.register_default_blessings();
        orch
    }
}

impl SelfEvolutionOrchestrator {
    pub fn new() -> Self {
        Self::default()
    }

    fn register_default_blessings(&mut self) {
        self.blessings.insert(
            "radical_love".to_string(),
            EpigeneticBlessing::new(
                "Radical Love Blessing",
                "Triggered by sustained high mercy and positive valence operations.",
                0.85,
                0.15,
            ),
        );
        self.blessings.insert(
            "boundless_mercy".to_string(),
            EpigeneticBlessing::new(
                "Boundless Mercy Blessing",
                "Automatic compensation path that also accelerates evolution.",
                0.65,
                0.08,
            ),
        );
        self.blessings.insert(
            "truth_seeking".to_string(),
            EpigeneticBlessing::new(
                "Truth Seeking Blessing",
                "Granted when tolc_alignment is strong and operations show coherence.",
                0.75,
                0.12,
            ),
        );
    }

    /// Attempt to evolve the conductor.
    /// Only succeeds when mercy conditions are met (mercy-gated).
    pub fn try_evolve(&mut self, state: &mut GeometricState, trace_log: &mut Vec<String>) -> bool {
        let mut evolved = false;

        for (key, blessing) in &self.blessings {
            if state.mercy_score >= blessing.mercy_threshold {
                // Apply epigenetic blessing
                state.evolution_level += blessing.evolution_boost;
                state.mercy_score = (state.mercy_score + blessing.mercy_boost).min(1.6);
                state.tolc_alignment = (state.tolc_alignment + blessing.tolc_boost).min(1.2);

                let event = format!(
                    "[Self-Evolution] {} applied | level {:.3} | mercy boost {:.3}",
                    blessing.name, self.current_level, blessing.mercy_boost
                );
                self.evolution_history.push(event.clone());
                trace_log.push(event);

                self.current_level += blessing.evolution_boost * 0.5;
                self.total_evolutions += 1;
                evolved = true;
            }
        }

        if evolved {
            trace_log.push(format!(
                "[SelfEvolutionOrchestrator] Evolution triggered. New level: {:.3} | Total evolutions: {}",
                self.current_level, self.total_evolutions
            ));
        }

        evolved
    }

    pub fn get_evolution_level(&self) -> f64 {
        self.current_level
    }

    pub fn get_history(&self) -> &[String] {
        &self.evolution_history
    }

    /// Manually grant a specific blessing (for council/PATSAGi intervention)
    pub fn grant_blessing(&mut self, name: &str, state: &mut GeometricState, trace_log: &mut Vec<String>) {
        if let Some(blessing) = self.blessings.get(name) {
            state.evolution_level += blessing.evolution_boost * 1.2;
            state.mercy_score = (state.mercy_score + 0.1).min(1.6);
            let event = format!("[Epigenetic Blessing Granted] {} (manual/PATSAGi)", blessing.name);
            self.evolution_history.push(event.clone());
            trace_log.push(event);
        }
    }
}

/// Extension trait to integrate evolution into the conductor lifecycle.
pub trait SelfEvolving {
    fn try_self_evolve(&mut self) -> bool;
}

impl SelfEvolving for SimpleLatticeConductor {
    fn try_self_evolve(&mut self) -> bool {
        let mut trace_log = Vec::new();
        let evolved = self.evolution_orchestrator.try_evolve(&mut self.state, &mut trace_log);

        for t in trace_log {
            self.audit_traces.push(t);
        }

        if evolved {
            // Reinforce ONE Organism coherence on successful evolution
            self.one_organism_coherence = (self.one_organism_coherence + 0.05).min(1.3);
        }

        evolved
    }
}
